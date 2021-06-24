from .constraints import *
from .modifiers import *
from .querybuilder import CB
from .snippets import SNIPPETS as S

import asyncio
import nest_asyncio
nest_asyncio.apply()

import multiprocessing
import pandas as pd
import subprocess
import hashlib
import time
import gzip
import glob
import json
import io


class Relation:
    _ridx = 1
    def __init__(self, evaluator, kind):
        self.evaluator = evaluator
        self.kind = kind
        self.arguments = []
        self.count_of = None
        self.ridx = Relation._ridx
        Relation._ridx += 1

    def args(self, *args):
        self.arguments = list(args)
        return self

    def process_arg(self, arg):
        if arg is None:
            return "_"

        return arg

    def get_args(self):
        return ", ".join([ self.process_arg(x) for x in self.arguments ])
    
    def make_count(self, label):
        self.count_of = label

    def __str__(self):
        if self.kind == '$eq':
            assert len(self.arguments) == 2
            return self.process_arg(self.arguments[0]) + ' = ' + self.process_arg(self.arguments[1])

        temp = self.kind + '(' + self.get_args() + ')' 
        
        if self.count_of is not None:
            return self.count_of + ' = count : { ' + temp + ' }'
        
        return temp

class IndexFilter:
    def __init__(self, the_words):
        self.the_words = the_words
    
    def __call__(self, file):
        with gzip.open(file, 'rb') as fh:
            as_json = json.load(fh)
            matches = []
            for word in self.the_words:
                matches.append(set(as_json[word] if word in as_json else []))
            return matches[0].intersection(*matches[1:])

class Evaluator:
    def __init__(self, query, should_debug=False):
        query.idify()
        
        self.worklist = [ query ]
        self.debug = should_debug

        self.outputs = []
        self.outputs_typed = []
        self.query = ''
        self.inputs = ''
        self.dataset = '/data/test-1k'
        self.labels = []

        self.pre_filter_words = []
        self.all_files = None

        self.query += str(
            Relation(self, "file_info").args("fid", "fpath")
        ) + ',\n'
        self.set_output("fid", "fpath")

    def build_query(self):
        last = None
        while len(self.worklist) > 0:
            root = self.worklist[-1]
            if len(root.children) == 0 or last is not None and (last in root.children):
                self.visit(root)
                self.worklist.pop()
                last = root
            else:
                for child in root.children[::-1]:
                    child.parent = root
                    self.worklist.append(child)

        header = self.get_header()
        body = self.get_body()
        footer = self.get_footer()

        if self.debug:
            print(header + body + footer)
            return False
        
        with open('/app/the-query.dl', 'w') as fh:
            fh.write(header + body + footer)
        
        return True
    
    async def get_files_from_word_index(self, words):
        index = pd.read_parquet(
            '{}/indices/text-to-files.parquet'.format(self.dataset)
        )

        sets = []
        for word in words:
            sets.append(set(index[index.text == word].file.unique()))
        
        all_files = sets[0].intersection(*sets)
        return all_files

    async def run(self, cmd):
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)

        stdout, stderr = await proc.communicate()
        return stdout, stderr

    async def compile_query(self):
        start = time.perf_counter()
        compile_result = await self.run(
            '/usr/local/bin/souffle -g- /app/the-query.dl'
        )

        # FIXUP
        the_program = compile_result[0].decode('utf-8')
        the_program = the_program.replace('return 0;\n', '\nobj.dumpOutputs();\nreturn 0;\n')
        
        with open('/app/the-query.cpp', 'w') as fh:
            fh.write(the_program)
        
        compile_result = await self.run(
            '/usr/local/bin/souffle-compile /app/the-query.cpp'
        )
        elapsed_time = time.perf_counter() - start

        if compile_result[1] != b'' and b'error' in compile_result[1]:
            print("Souffle compile error")
            print(compile_result[1])
            return False

        print(f"  + Compile time: {elapsed_time:.4f}s")
        return True
    
    async def select_files(self, limit=None):
        start = time.perf_counter()
        files = []
    
        
        if len(self.pre_filter_words) > 0:
            files = await self.get_files_from_word_index(
                self.pre_filter_words
            )
            if self.debug:
                print("Retrived files from index (words pre-filter)...")
                print("  + Found {} files".format(len(files)))
        else:
            self.all_files = []
            with open('{}/indices/all-files.txt'.format(self.dataset), 'r') as fh:
                text = fh.read()
                for line in text.split('\n'):
                    if len(line.strip()) > 0:
                        self.all_files.append(line.strip())
            self.all_files = sorted(self.all_files)
            files = self.all_files
            if self.debug:
                print("Retrived all files (no index)...")
                print("  + Found {} files".format(len(files)))
        
        if self.debug and limit is not None:
            print("Limiting to {} files".format(limit))
        elapsed_time = time.perf_counter() - start

        print(f"  + File select time: {elapsed_time:.4f}s")
        
        if limit is not None:
            return files[:limit]
        
        return files

    def eval(self, compile=False):
        total_start = time.perf_counter()

        # Build the query first
        if not self.build_query():
            return pd.DataFrame()

        # Maybe compile
        in_file = '/app/the-query.dl'
        targets = []
        if compile:
            status, targets = asyncio.run(asyncio.gather(
                self.compile_query(),
                self.select_files()
            ))
            # Compile failed?
            if not status:
                return
            in_file = in_file[:-3] # Remove the .dl
        else: 
            targets = asyncio.run(self.select_files())

        start = time.perf_counter()
        exec_result = subprocess.run([
            'bash',
            '-c',
            'parallel -k ' + (
                "souffle -F{}.flow/ -D- /app/the-query.dl" if not compile 
                else "/app/the-query -F{}.flow/"
            )
        ], input=('\n'.join(targets) + '\n').encode('utf-8'), capture_output=True)
        elapsed_time = time.perf_counter() - start
        print(f"  + Query time: {elapsed_time:.4f}s")


        start = time.perf_counter()
        good_text = ''
        decoded = exec_result.stdout.decode().split('\n')
        for line in decoded:
            if '\t' not in line or line == '\t'.join(self.outputs):
                continue
            good_text += line + '\n'

        final = pd.read_csv(io.BytesIO(
            good_text.encode('utf-8')
        ), sep='\t', names=self.outputs)
        elapsed_time = time.perf_counter() - start
        print(f"  + Collation time: {elapsed_time:.4f}s")

        elapsed_time = time.perf_counter() - total_start
        print(f"Total time: {elapsed_time:.4f}s")

        return final

    def get_body(self):
        return self.query.strip().rstrip(',') + '\n  .\n'

    def get_footer(self):
        footer = ".output synth_query\n"
        return footer

    def get_header(self):
        final = ""
        final += '#include "/app/applications/jupyter-extension/nteract_on_jupyter/notebooks/codebook/python/snippets/prelude.dl"\n'
        final += self.inputs
        final += ".decl synth_query(" + ", ".join(self.outputs_typed) + ")\n"
        final +=  "synth_query(" + ", ".join(self.outputs) +") :- \n"
        return final

    def set_output(self, *args):
        for arg in list(args):
            if arg in self.outputs:
                continue
            self.outputs.append(arg)
            if arg.startswith('start_byte_') or arg.startswith('end_byte_') or arg.startswith('count_'):
                self.outputs_typed.append(arg + ":number")
            elif arg.startswith('start_line_') or arg.startswith('end_line_'):
                self.outputs_typed.append(arg + ":number")
            elif arg.startswith('start_col_') or arg.startswith('end_col_'):
                self.outputs_typed.append(arg + ":number")
            elif arg.startswith('source_text_'):
                self.outputs_typed.append(arg + ":SourceText")
            elif arg.startswith('gid_'):
                self.outputs_typed.append(arg + ":Gid")
            elif arg == "fid":
                self.outputs_typed.append(arg + ':Fid')
            elif arg == "fpath":
                self.outputs_typed.append(arg + ':symbol')
            elif arg.startswith('out_name'):
                self.outputs_typed.append(arg + ':symbol')
            elif arg.startswith('out_module_name'):
                self.outputs_typed.append(arg + ':symbol')
            else:
                self.outputs_typed.append(arg + ':symbol')

  
    def visit(self, node):
        name_var = None
        if 'name' in node.selects:
            text, outputs = S['/constraints/name.dl'](node)
            name_var = outputs['name']
            self.query += text

            if node.select_as:
                self.query += str(
                    Relation(self, '$eq').args(node.select_as, name_var)
                ) + ',\n'
                self.set_output(node.select_as)
            else:
                self.set_output(name_var)
        if 'module_name' in node.selects:
            text, outputs = S['/constraints/module_name.dl'](node)
            module_name_var = outputs['module_name']
            self.query += text

            if node.select_as:
                self.query += str(
                    Relation(self, '$eq').args(node.select_as, module_name_var)
                ) + ',\n'
                self.set_output(node.select_as)
            else:
                self.set_output(module_name_var)
        if 'text' in node.selects:
            text, outputs = S['/constraints/text.dl'](node)
            text_var = outputs['text']
            self.query += text

            if node.select_as:
                self.query += str(
                    Relation(self, '$eq').args(node.select_as, text_var)
                ) + ',\n'
                self.set_output(node.select_as)
            else:
                self.set_output(text_var)
        
        if node.type is not None and node.type not in ['_', '$use', '$df']:
            tlabel = 'type_{}'.format(node.id if node.label is None else node.label)
            type_cons = node.type if isinstance(node.type, list) else [ node.type ]
            self.query += '({}),\n'.format(
                ' ; '.join(
                    [ '{} = "{}"'.format(tlabel, x) for x in type_cons ]
                )
            )
        
        if node.label is not None and node.label not in self.labels: 
            self.labels.append(node.label)
            self.query += S['/select_node.dl'](node)
            self.set_output(
                'source_text_{}'.format(node.label),
                'start_line_{}'.format(node.label),
                'start_col_{}'.format(node.label),
                'end_line_{}'.format(node.label),
                'end_col_{}'.format(node.label),
                'gid_{}'.format(node.label)
            )
        elif node.type is not None and node.label not in self.labels:
            self.query += S['/node.dl'](node)

        if node.type == '$use':
            self.query += S['/modifiers/use.dl'](node)
        if node.type == "$df":
            self.query += S['/modifiers/df.dl'](node)
        if node.has_mod(CBCallTarget):
            self.query += S['/modifiers/call_target.dl'](node)
        if node.has_mod(CBModuleRoot):
            self.query += S['/modifiers/module_root.dl'](node)
        if node.has_mod(CBAnyParentIs):
            self.query += S['/modifiers/any_parent.dl'](node)
        if node.has_mod(CBAnyChildIs):
            self.query += S['/modifiers/any_child.dl'](node)
        if node.has_mod(CBRhs):
            self.query += S['/modifiers/the_rhs.dl'](node)
        if node.has_mod(CBLhs):
            self.query += S['/modifiers/the_lhs.dl'](node)
        if node.has_mod(CBObjectIs):
            self.query += S['/modifiers/the_object.dl'](node)
        if node.has_mod(CBAttributeIs):
            self.query += S['/modifiers/the_attribute.dl'](node)
        if node.has_mod(CBSubscriptIs):
            self.query += S['/modifiers/the_subscript.dl'](node)
        if node.has_mod(CBOnlySubscriptIs):
            self.query += S['/modifiers/the_only_subscript.dl'](node)
        if node.has_mod(CBValueIs):
            self.query += S['/modifiers/the_value.dl'](node)
        if node.has_mod(CBChildIs):
            self.query += S['/modifiers/child.dl'](node)
        if node.has_mod(CBFirstChildIs):
            self.query += S['/modifiers/first_child.dl'](node)
        if node.has_mod(CBSecondChildIs):
            self.query += S['/modifiers/second_child.dl'](node)
        if node.has_mod(CBThirdChildIs):
            self.query += S['/modifiers/third_child.dl'](node)
        if node.has_mod(CBNoThirdChild):
            self.query += S['/modifiers/no_third_child.dl'](node)
        
        # If we have a constraint on text, add that
        con = node.get_con(CBText)
        if con is not None:
            text, outputs = S['/constraints/text.dl'](node)
            self.query += text + str(
                Relation(self, "$eq").args(outputs['text'], '"{}"'.format(con.value))
            ) + ',\n'
            self.pre_filter_words.append(con.value)

        # If we have the same text as some other part of the query,
        # enforce that constraint
        con = node.get_con(CBSameText)
        if con is not None:
            text, outputs = S['/constraints/text.dl'](node)
            self.query += text + str(
                Relation(self, "$eq").args(outputs['text'], 'source_text_{}'.format(con.value))
            ) + ',\n'
        
        # If we have a constraint on name, add that 
        con = node.get_con(CBName)
        if con is not None:
            if name_var is None:
                text, outputs = S['/constraints/name.dl'](node)
                name_var = outputs['name']
                self.query += text
            self.query += str(
                Relation(self, '$eq').args(name_var, '"{}"'.format(con.value))
            ) + ',\n'
            self.pre_filter_words.append(con.value)

        # If we have a constraint on ids, add that 
        con = node.get_con(CBFromSet)
        if con is not None:
            glabel = 'gid_{}'.format(node.id if node.label is None else node.label)
            self.inputs += ".decl input_" + str(node.id) + '(gid:Gid)\n' + \
                ".input input_" + str(node.id) + \
                '(IO=file, filename="{}", delimiter="\\t")\n\n'.format(con.file)
            self.query += str(
                Relation(self, 'input_' + str(node.id)).args(glabel)
            ) + ',\n'

        # If we have a constraint on child type, add that 
        con = node.get_con(CBEveryChildHasType)
        if con is not None:
            text, outputs = S['/constraints/every_child_has_type.dl'](node)
            child_type_var = outputs['child_type']
            self.query += text
            tlabel = 'type_{}'.format(node.id if node.label is None else node.label)
            self.query += str(
                Relation(self, '$eq').args(child_type_var, '"{}"'.format(con.value))
            ) + ',\n'


