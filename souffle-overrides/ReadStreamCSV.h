/*
 * Souffle - A Datalog Compiler
 * Copyright (c) 2021, The Souffle Developers. All rights reserved
 * Licensed under the Universal Permissive License v 1.0 as shown at:
 * - https://opensource.org/licenses/UPL
 * - <souffle root>/licenses/SOUFFLE-UPL.txt
 */

/************************************************************************
 *
 * @file ReadStreamCSV.h
 *
 ***********************************************************************/

#pragma once

#include "souffle/RamTypes.h"
#include "souffle/SymbolTable.h"
#include "souffle/io/ReadStream.h"
#include "souffle/utility/ContainerUtil.h"
#include "souffle/utility/FileUtil.h"
#include "souffle/utility/StringUtil.h"

#ifdef USE_LIBZ
#include "souffle/io/gzfstream.h"
#else
#include <fstream>
#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace souffle {
class RecordTable;

class ReadStreamCSV : public ReadStream {
public:
    ReadStreamCSV(std::istream& file, const std::map<std::string, std::string>& rwOperation,
            SymbolTable& symbolTable, RecordTable& recordTable)
            : ReadStream(rwOperation, symbolTable, recordTable),
              rfc4180(getOr(rwOperation, "rfc4180", "false") == std::string("true")),
              delimiter(getOr(rwOperation, "delimiter", (rfc4180 ? "," : "\t"))), file(file), lineNumber(0),
              inputMap(getInputColumnMap(rwOperation, static_cast<unsigned int>(arity))) {
        if (rfc4180 && delimiter.find('"') != std::string::npos) {
            std::stringstream errorMessage;
            errorMessage << "CSV delimiter cannot contain '\"' character when rfc4180 is enabled.";
            throw std::invalid_argument(errorMessage.str());
        }

        while (inputMap.size() < arity) {
            int size = static_cast<int>(inputMap.size());
            inputMap[size] = size;
        }
    }

public:
    /**
     * Read and return the next tuple.
     *
     * Returns nullptr if no tuple was readable.
     * @return
     */
    Own<RamDomain[]> readNextTuple() override {
        if (file.eof()) {
            return nullptr;
        }
        std::string line;
        Own<RamDomain[]> tuple = mk<RamDomain[]>(typeAttributes.size());

        if (!getline(file, line)) {
            return nullptr;
        }
        // Handle Windows line endings on non-Windows systems
        if (!line.empty() && line.back() == '\r') {
            line = line.substr(0, line.length() - 1);
        }
        ++lineNumber;

        std::size_t start = 0;
        std::size_t columnsFilled = 0;
        for (uint32_t column = 0; columnsFilled < arity; column++) {
            std::size_t charactersRead = 0;
            std::string element = nextElement(line, start);
            if (inputMap.count(column) == 0) {
                continue;
            }
            ++columnsFilled;

            try {
                auto&& ty = typeAttributes.at(inputMap[column]);
                switch (ty[0]) {
                    case 's': {
                        tuple[inputMap[column]] = symbolTable.unsafeEncode(element);
                        charactersRead = element.size();
                        break;
                    }
                    case 'r': {
                        tuple[inputMap[column]] = readRecord(element, ty, 0, &charactersRead);
                        break;
                    }
                    case '+': {
                        tuple[inputMap[column]] = readADT(element, ty, 0, &charactersRead);
                        break;
                    }
                    case 'i': {
                        tuple[inputMap[column]] = RamSignedFromString(element, &charactersRead);
                        break;
                    }
                    case 'u': {
                        tuple[inputMap[column]] = ramBitCast(readRamUnsigned(element, charactersRead));
                        break;
                    }
                    case 'f': {
                        tuple[inputMap[column]] = ramBitCast(RamFloatFromString(element, &charactersRead));
                        break;
                    }
                    default: fatal("invalid type attribute: `%c`", ty[0]);
                }
                // Check if everything was read.
                if (charactersRead != element.size()) {
                    throw std::invalid_argument(
                            "Expected: " + delimiter + " or \\n. Got: " + element[charactersRead]);
                }
            } catch (...) {
                std::stringstream errorMessage;
                errorMessage << "Error converting <" + element + "> in column " << column + 1 << " in line "
                             << lineNumber << "; ";
                throw std::invalid_argument(errorMessage.str());
            }
        }

        return tuple;
    }

    /**
     * Read an unsigned element. Possible bases are 2, 10, 16
     * Base is indicated by the first two chars.
     */
    RamUnsigned readRamUnsigned(const std::string& element, std::size_t& charactersRead) {
        // Sanity check
        assert(element.size() > 0);

        RamSigned value = 0;

        // Check prefix and parse the input.
        if (isPrefix("0b", element)) {
            value = RamUnsignedFromString(element, &charactersRead, 2);
        } else if (isPrefix("0x", element)) {
            value = RamUnsignedFromString(element, &charactersRead, 16);
        } else {
            value = RamUnsignedFromString(element, &charactersRead);
        }
        return value;
    }

    std::string nextElement(const std::string& line, std::size_t& start) {
        std::string element;

        if (rfc4180) {
            if (line[start] == '"') {
                // quoted field
                const std::size_t end = line.length();
                std::size_t pos = start + 1;
                bool foundEndQuote = false;
                while (pos < end) {
                    char c = line[pos++];
                    if (c == '"' && (pos < end) && line[pos] == '"') {
                        // two double-quote => one double-quote
                        element.push_back('"');
                        ++pos;
                    } else if (c == '"') {
                        foundEndQuote = true;
                        break;
                    } else {
                        element.push_back(c);
                    }
                }

                if (!foundEndQuote) {
                    // missing closing quote
                    std::stringstream errorMessage;
                    errorMessage << "Unbalanced field quote in line " << lineNumber << "; ";
                    throw std::invalid_argument(errorMessage.str());
                }

                // field must be immediately followed by delimiter or end of line
                if (pos != line.length()) {
                    std::size_t nextDelimiter = line.find(delimiter, pos);
                    if (nextDelimiter != pos) {
                        std::stringstream errorMessage;
                        errorMessage << "Separator expected immediately after quoted field in line "
                                     << lineNumber << "; ";
                        throw std::invalid_argument(errorMessage.str());
                    }
                }

                start = pos + delimiter.size();
                return element;
            } else {
                // non-quoted field, span until next delimiter or end of line
                const std::size_t end = std::min(line.find(delimiter, start), line.length());
                element = line.substr(start, end - start);
                start = end + delimiter.size();

                return element;
            }
        }

        std::size_t end = start;
        // Handle record/tuple delimiter coincidence.
        if (delimiter.find(',') != std::string::npos) {
            int record_parens = 0;
            std::size_t next_delimiter = line.find(delimiter, start);

            // Find first delimiter after the record.
            while (end < std::min(next_delimiter, line.length()) || record_parens != 0) {
                // Track the number of parenthesis.
                if (line[end] == '[') {
                    ++record_parens;
                } else if (line[end] == ']') {
                    --record_parens;
                }

                // Check for unbalanced parenthesis.
                if (record_parens < 0) {
                    break;
                };

                ++end;

                // Find a next delimiter if the old one is invalid.
                // But only if inside the unbalance parenthesis.
                if (end == next_delimiter && record_parens != 0) {
                    next_delimiter = line.find(delimiter, end);
                }
            }

            // Handle the end-of-the-line case where parenthesis are unbalanced.
            if (record_parens != 0) {
                std::stringstream errorMessage;
                errorMessage << "Unbalanced record parenthesis in line " << lineNumber << "; ";
                throw std::invalid_argument(errorMessage.str());
            }
        } else {
            end = std::min(line.find(delimiter, start), line.length());
        }

        // Check for missing value.
        if (start > end) {
            std::stringstream errorMessage;
            errorMessage << "Values missing in line " << lineNumber << "; ";
            throw std::invalid_argument(errorMessage.str());
        }

        element = line.substr(start, end - start);
        start = end + delimiter.size();

        return element;
    }

    std::map<int, int> getInputColumnMap(
            const std::map<std::string, std::string>& rwOperation, const unsigned arity_) const {
        std::string columnString = getOr(rwOperation, "columns", "");
        std::map<int, int> inputColumnMap;

        if (!columnString.empty()) {
            std::istringstream iss(columnString);
            std::string mapping;
            int index = 0;
            while (std::getline(iss, mapping, ':')) {
                inputColumnMap[stoi(mapping)] = index++;
            }
            if (inputColumnMap.size() < arity_) {
                throw std::invalid_argument("Invalid column set was given: <" + columnString + ">");
            }
        } else {
            while (inputColumnMap.size() < arity_) {
                int size = static_cast<int>(inputColumnMap.size());
                inputColumnMap[size] = size;
            }
        }
        return inputColumnMap;
    }

    const bool rfc4180;
    const std::string delimiter;
    std::istream& file;
    std::size_t lineNumber;
    std::map<int, int> inputMap;
};

class ReadFileCSV : public ReadStreamCSV {
public:
    ReadFileCSV(const std::map<std::string, std::string>& rwOperation, SymbolTable& symbolTable,
            RecordTable& recordTable)
            : ReadStreamCSV(fileHandle, rwOperation, symbolTable, recordTable),
              baseName(souffle::baseName(getFileName(rwOperation))),
              fileHandle(getFileName(rwOperation), std::ios::in | std::ios::binary) {
        if (!fileHandle.is_open()) {
            throw std::invalid_argument("Cannot open fact file " + baseName + "\n");
        }
        // Strip headers if we're using them
        if (getOr(rwOperation, "headers", "false") == "true") {
            std::string line;
            getline(file, line);
        }
    }

    /**
     * Read and return the next tuple.
     *
     * Returns nullptr if no tuple was readable.
     * @return
     */
    Own<RamDomain[]> readNextTuple() override {
        try {
            return ReadStreamCSV::readNextTuple();
        } catch (std::exception& e) {
            std::stringstream errorMessage;
            errorMessage << e.what();
            errorMessage << "cannot parse fact file " << baseName << "!\n";
            throw std::invalid_argument(errorMessage.str());
        }
    }

    ~ReadFileCSV() override = default;

protected:
    /**
     * Return given filename or construct from relation name.
     * Default name is [configured path]/[relation name].facts
     *
     * @param rwOperation map of IO configuration options
     * @return input filename
     */
    static std::string getFileName(const std::map<std::string, std::string>& rwOperation) {
        auto name = getOr(rwOperation, "filename", rwOperation.at("name") + ".facts");
        if (name.front() != '/') {
            name = getOr(rwOperation, "fact-dir", ".") + "/" + name;
        }
        return name;
    }

    std::string baseName;
#ifdef USE_LIBZ
    gzfstream::igzfstream fileHandle;
#else
    std::ifstream fileHandle;
#endif
};

class ReadCinCSVFactory : public ReadStreamFactory {
public:
    Own<ReadStream> getReader(const std::map<std::string, std::string>& rwOperation, SymbolTable& symbolTable,
            RecordTable& recordTable) override {
        return mk<ReadStreamCSV>(std::cin, rwOperation, symbolTable, recordTable);
    }

    const std::string& getName() const override {
        static const std::string name = "stdin";
        return name;
    }
    ~ReadCinCSVFactory() override = default;
};

class ReadFileCSVFactory : public ReadStreamFactory {
public:
    Own<ReadStream> getReader(const std::map<std::string, std::string>& rwOperation, SymbolTable& symbolTable,
            RecordTable& recordTable) override {
        return mk<ReadFileCSV>(rwOperation, symbolTable, recordTable);
    }

    const std::string& getName() const override {
        static const std::string name = "file";
        return name;
    }

    ~ReadFileCSVFactory() override = default;
};









class ReadFilesCSV : public ReadStream  {
public:
    ReadFilesCSV(const std::map<std::string, std::string>& rwOperation, SymbolTable& symbolTable,
            RecordTable& recordTable) : 
            files(get_files(rwOperation)),
            ReadStream(rwOperation, symbolTable, recordTable) {

        for (int i = 0; i < files.size(); ++i) {
            streams.push_back(
                ReadStreamCSV(
                    *files[i],
                    rwOperation,
                    symbolTable,
                    recordTable
                )
            );
        }
    }

    /**
     * Read and return the next tuple.
     *
     * Returns nullptr if no tuple was readable.
     * @return
     */
    Own<RamDomain[]> readNextTuple() override {
        try {
            if (fidx >= streams.size()) {
                return nullptr;
            } else {
                auto res = streams[fidx].readNextTuple();
                if (res == nullptr) {
                    fidx += 1;
                    return readNextTuple();
                }
                return res;
            }
        } catch (std::exception& e) {
            std::stringstream errorMessage;
            errorMessage << e.what();
            errorMessage << "cannot parse one or more fact files!\n";
            throw std::invalid_argument(errorMessage.str());
        }
    }

    ~ReadFilesCSV() {
        for (int i = 0; i < files.size(); ++i) {
            delete files[i];
        }
    }

protected:
    static std::vector<gzfstream::igzfstream*> get_files(const std::map<std::string, std::string>& rwOperation) {
        auto name = getOr(rwOperation, "filename", rwOperation.at("name") + ".facts");
        std::string dirs = getOr(rwOperation, "fact-dir", "");
        std::vector<gzfstream::igzfstream*> temp;
        std::ifstream indirs(dirs);
        std::string temporary;
        while (std::getline(indirs, temporary)) {
            if (!std::all_of(temporary.begin(), temporary.end(), isspace)) {
                std::string fullname = temporary + '/' + name;
                // std::cout << "open: " << fullname << std::endl;
                temp.push_back(
                    new gzfstream::igzfstream(fullname)
                );
            }
        }

        return temp;
    }

    std::string baseName;
    int32_t fidx = 0;
    std::vector<gzfstream::igzfstream*> files;
    std::vector<ReadStreamCSV> streams;
};

class ReadFilesCSVFactory : public ReadStreamFactory {
public:
    Own<ReadStream> getReader(const std::map<std::string, std::string>& rwOperation, SymbolTable& symbolTable,
            RecordTable& recordTable) override {
        return mk<ReadFilesCSV>(rwOperation, symbolTable, recordTable);
    }

    const std::string& getName() const override {
        static const std::string name = "files";
        return name;
    }

    ~ReadFilesCSVFactory() override = default;
};

} /* namespace souffle */