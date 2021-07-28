from .constraints import *
from .modifiers import *
from .querybuilder import CB
from .snippets import SNIPPETS as S, refresh_snippet_offsets

import sys
sys.path.append('/arrow/python')


import multiprocessing
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import subprocess
import sqlite3
import hashlib
import xxhash
import time
import gzip
import glob
import tqdm
import json
import io
import os


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



def hashit(thing):
    return int.from_bytes(
        xxhash.xxh64(thing, seed=3235823838).intdigest().to_bytes(8, byteorder='little'),
        signed=True, byteorder="little"
    )


class ChunkMeta:
    def __init__(self, chunk, rindex, fpaths):
        self.chunk = chunk
        self.rindex = rindex
        self.fpaths = fpaths
        self.fids_to_fpaths = {
            hashit('parsed/' + os.path.basename(fpath) + '.xml.gz'): fpath for fpath in self.fpaths
        }
        self.fids_set = set(self.rindex.fid.unique())
    
    def write_out_filter(self, hashed_words, allowable_fids=None):
        # If we have no constraints, we write no file
        if len(hashed_words) == 0 and allowable_fids is None:
            try:
                os.remove('/tmp/query-inputs{}/relations/filter-fids.txt'.format(self.chunk))
            except:
                pass
            return len(self.fids_set)

        # Matches ALL the prefilter words
        matches = [ self.fids_set ]
        for word in hashed_words:
            matches.append(set(
                self.rindex[self.rindex.tid == word].fid.unique()
            ))
        
        matches = matches[0].intersection(*matches)

        # If we are further constrained, write that out
        if allowable_fids is not None:
            matches = matches.intersection(allowable_fids)
        
        os.makedirs('/tmp/query-inputs{}/relations'.format(self.chunk), exist_ok=True)
        os.makedirs('/tmp/query-outputs{}/relations'.format(self.chunk), exist_ok=True)

        with open('/tmp/query-inputs{}/relations/filter-fids.txt'.format(self.chunk), 'w') as fh:
            for match in matches:
                fh.write(str(match) + '\n')
        
        return len(matches)


class MetaFilter:
    def __init__(self, hashed_words, allowable_fids=None):
        self.hashed_words = hashed_words
        self.allowable_fids = allowable_fids
    
    def __call__(self, chunk):
        return chunk.write_out_filter(
            self.hashed_words, self.allowable_fids
        )


class IndexFilter:
    def __init__(self):
        self.rindex = None
        self.chunk = ""
    
    def __call__(self, chunk):
        rindex = pd.read_parquet(
            '{}/relations/rindex.merged'.format(chunk),
            columns=['tid', 'fid']
        )
        
        fpaths = []
        with open('{}/listing.txt'.format(chunk)) as fh:
            fpaths = [ x.strip() for x in fh.readlines() if len(x.strip()) > 0 ]
        
        return ChunkMeta(chunk, rindex, fpaths)



class Evaluator:
    _dataset = '/data/test-1k'

    @staticmethod
    def resolve_fid(fid):
        return Evaluator._fids_to_fpaths[fid]

    @staticmethod
    def use_ds_test_1k():
        Evaluator._dataset = '/data/test-1k'
    
    @staticmethod
    def use_ds_gh_2017():
        index_filter = IndexFilter()

        Evaluator._dataset = '/data/gh-2017'

        # pool = multiprocessing.Pool()
        # chunks = list(glob.glob(Evaluator._dataset + '/chunked/chunk-*'))

        # print('Using dataset "{}"'.format(Evaluator._dataset))
        # print('  + Dataset has {} chunks'.format(len(chunks)))

        # for chunk in tqdm.tqdm(pool.imap(index_filter, chunks), desc="  + Loading metadata", total=len(chunks)):
        #     Evaluator._chunks_meta.append(chunk)
        #     for fid, fpath in chunk.fids_to_fpaths.items():
        #         Evaluator._fids_to_fpaths[fid] = fpath

    @staticmethod
    def use_ds_gh_2019():
        Evaluator._dataset = '/data/gh-2019'
    
    @staticmethod
    def create_sqlite_db(force=False):
        if os.path.isfile(Evaluator._dataset + '/metadata.sqlite.db') and not force:
            return

        con = sqlite3.connect(Evaluator._dataset + '/metadata.sqlite.db')

        index_filter = IndexFilter()
        pool = multiprocessing.Pool()
        chunks = list(glob.glob(Evaluator._dataset + '/chunked/chunk-*'))

        cur = con.cursor()

        cur.execute('CREATE TABLE chunks_fids_paths (chunk text, fid Int64, path text)')
        cur.execute('CREATE TABLE selected_fids (fid Int64)')

        print('[One time setup] Using dataset "{}"'.format(Evaluator._dataset))
        print('  + Dataset has {} chunks'.format(len(chunks)))

        for chunk in tqdm.tqdm(pool.imap(index_filter, chunks), desc="  + Loading metadata", total=len(chunks)):
            values = []
            for fid, fpath in chunk.fids_to_fpaths.items():
                values.append((chunk.chunk, fid, fpath))
            cur.executemany("INSERT INTO chunks_fids_paths VALUES (?, ?, ?)", values)
            
        con.commit()
        con.close()
    
    @staticmethod
    def use_ds_gh_2020():
        Evaluator._dataset = '/data/gh-2020'

    def __init__(self, query, should_debug=False, prefilter_files=None):
        query.idify()
        self.inid = 1
        
        self.worklist = [ query ]
        self.debug = should_debug

        self.outputs = []
        self.outputs_typed = []
        self.query = []
        self.inputs = ''
        self.dataset = Evaluator._dataset
        self.meta_db = sqlite3.connect(self.dataset + '/metadata.sqlite.db')
        self.labels = []

        self.pre_filter_words = []
        self.all_files = None

        self.query_dl = None
        self.files_prefilters = []

        if prefilter_files is not None:
            self.files_prefilters.append(set(prefilter_files))

        # self.query.append((str(
        #     Relation(self, "file_info").args("fid", "fpath")
        # ) + ',\n', 1))
        self.set_output("fid")

    def build_query(self, compiled):
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

        header = self.get_header(compiled)
        body = self.get_body()
        footer = self.get_footer()

        if self.debug:
            print(header + body + footer)
            return False

        extra_to_hash = ""
        prelude_prefix = "/app/applications/jupyter-extension/nteract_on_jupyter/notebooks/codebook/python/snippets/"
        with open(prelude_prefix + "/prelude.dl", "r") as fh:
            extra_to_hash += fh.read()
        with open(prelude_prefix + "/utils.dl", "r") as fh:
            extra_to_hash += fh.read()

        query_hash = hashlib.sha256(
            (header + body + footer + extra_to_hash).encode('utf-8')
        ).hexdigest()
        self.query_dl = '/tmp/queries/{}.dl'.format(query_hash) 
        self.query_prof_dl = '/tmp/queries/{}.prof.dl'.format(query_hash) 

        with open(self.query_dl, 'w') as fh:
            fh.write(header + body + footer)
        with open(self.query_prof_dl, 'w') as fh:
            fh.write(self.get_header(False) + body + footer)
        
        return True
    
    def get_files_from_word_index(self, words):
        files =[ set(pd.read_parquet(
            '{}/indices/text-to-files'.format(self.dataset),
            filters=[('text', '=', word)],
            columns=['file']
        ).file.unique()) for word in words ]

        return set.intersection(*files)

    def compile_query(self): 
        if os.path.isfile('{}.cpp'.format(self.query_dl[:-3])):
            print('  + Query already compiled (cached) `{}`'.format(self.query_dl))
            return True
        
        start = time.perf_counter()

        profile_result = subprocess.run([
            '/app/souffle.sh',
            '-S',
            '-p',
            '{}'.format(self.query_prof_dl.replace('.dl', '')),
            '--profile-frequency',
            '-F',
            self.dataset + '/chunked/chunk-1553/relations',
            '{}'.format(self.query_prof_dl),
        ], capture_output=True)

        elapsed_time = time.perf_counter() - start
        print(f"  + Profile time: {elapsed_time:.4f}s")


        start = time.perf_counter()
        compile_result = subprocess.run([
            '/app/souffle.sh',
            '-S',
            '-PSIPS:profile-use',
            '-u',
            '{}'.format(self.query_prof_dl.replace('.dl', '')),
            '-g-',
            '{}'.format(self.query_dl)
        ], capture_output=True)

        the_program = compile_result.stdout.decode('utf-8')

        # NOTE: below is not needed now that we are back to 
        # writing to disk / reading back in for collation

        # FIXUP
        # the_program = the_program.replace('return 0;\n', '\nobj.dumpOutputs();\nreturn 0;\n')
        
        with open('{}.cpp'.format(self.query_dl[:-3]), 'w') as fh:
            fh.write(the_program)

        compile_result = subprocess.run([
            '/usr/local/bin/souffle-compile',
            '{}.cpp'.format(self.query_dl[:-3])
        ], capture_output=True)
        elapsed_time = time.perf_counter() - start

        if compile_result.stderr != b'' and b'error' in compile_result.stderr:
            print("Souffle compile error")
            print(compile_result.stderr)
            return False

        print(f"  + Compile time: {elapsed_time:.4f}s")
        return True
    
    def select_files(self, limit=None):
        # So, in the new version of this, I'm thinking we write out a file per chunk
        # that has the fids we have downfiltered to OR, if no filtering applies, we 
        # don't write out any files

        assert limit is None, "Limiting not supported currently"

        start = time.perf_counter()

        allowable = None
        if len(self.files_prefilters) > 0:
            allowable = set.intersection(*self.files_prefilters)
            print("  + Had only {} allowable files (pre-filter files)".format(len(allowable)))
        
        print('  - TODO: re-enable writing pre-filters.')
        # cur = self.meta_db.cursor()
        # cur.execute('DELETE FROM selected_fids')

        # if allowable is not None:
        #     cur.executemany('INSERT INTO selected_fids VALUES (?)', list(allowable))
        
        # self.meta_db.commit()

        elapsed_time = time.perf_counter() - start

        print(f"  + File select time: {elapsed_time:.4f}s")
        # print("  + Found {} matching files".format(sum(counts)))

    def eval(self, compile=False):
        refresh_snippet_offsets()

        total_start = time.perf_counter()

        # Build the query first
        if not self.build_query(compile):
            return pd.DataFrame()

        # Maybe compile
        in_file = self.query_dl
        if compile:
            status = self.compile_query()
            # Compile failed?
            if not status:
                return
            in_file = in_file[:-3] # Remove the .dl

        self.select_files()

        targets = list(glob.glob(self.dataset + '/chunked/chunk-*/relations'))
        start = time.perf_counter()

        out_files = list(
            glob.glob('/tmp/query-outputs' + self.dataset + '/chunked/chunk-*/relations/results/data.parquet')
        )

        for file in out_files:
            os.remove(file)

        #  -D/tmp/query-outputs/{}
        exec_result = subprocess.run([
            'bash',
            '-c',
            'parallel -k ' + (
                ("/app/souffle.sh -F{} -D/tmp/query-outputs/{} " + in_file) if not compile 
                else ("/app/wrapper.sh " + in_file + " -F{} -D/tmp/query-outputs/{}")
            )
        ], input=('\n'.join(targets) + '\n').encode('utf-8'), capture_output=True)
        elapsed_time = time.perf_counter() - start
        print(f"  + Query time: {elapsed_time:.4f}s")

        start = time.perf_counter()

        out_files = list(
            glob.glob('/tmp/query-outputs' + self.dataset + '/chunked/chunk-*/relations/results/data.parquet')
        )

        final = pq.ParquetDataset(
            out_files
        ).read().to_pandas()

        elapsed_time = time.perf_counter() - start
        print(f"  + Collation time: {elapsed_time:.4f}s")

        elapsed_time = time.perf_counter() - total_start
        print(f"Total time: {elapsed_time:.4f}s")

        return final

    def get_body(self):
        clauses = sorted(self.query, key=lambda x:x[1])
        query = ''.join([ x[0] for x in clauses ])
        return query.strip().rstrip(',') + '\n  .\n'

    def get_footer(self):
        footer = ".output synth_query(IO=parquet, filename=\"results\")\n"
        return footer

    def get_header(self, compiled):
        final = ""
        if compiled:
            final += '#include "/app/applications/jupyter-extension/nteract_on_jupyter/notebooks/codebook/python/snippets/prelude.dl"\n'
        else:
            final += '#include "/app/applications/jupyter-extension/nteract_on_jupyter/notebooks/codebook/python/snippets/interp-prelude.dl"\n'
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
            elif arg.startswith('out_to_'):
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
            text, outputs, cost = S['/constraints/name.dl'](node)
            name_var = outputs['name']
            self.query.append((text, cost))

            if node.select_as:
                self.query.append((str(
                    Relation(self, '$eq').args(node.select_as, name_var)
                ) + ',\n', 1))
                self.set_output(node.select_as)
            else:
                self.set_output(name_var)
        if 'module_name' in node.selects:
            text, outputs, cost = S['/constraints/module_name.dl'](node)
            module_name_var = outputs['module_name']
            self.query.append((text, cost))

            if node.select_as:
                self.query.append((str(
                    Relation(self, '$eq').args(node.select_as, module_name_var)
                ) + ',\n', 1))
                self.set_output(node.select_as)
            else:
                self.set_output(module_name_var)
        if 'text' in node.selects:
            text, outputs, cost = S['/constraints/text.dl'](node)
            text_var = outputs['text']
            self.query.append((text, cost))

            if node.select_as:
                self.query.append((str(
                    Relation(self, '$eq').args(node.select_as, text_var)
                ) + ',\n', 1))
                self.set_output(node.select_as)
            else:
                self.set_output(text_var)
        
        if node.type is not None and node.type not in ['_', '$use', '$df', '$literal']:
            tlabel = 'type_{}'.format(node.id if node.label is None else node.label)
            type_cons = node.type if isinstance(node.type, list) else [ node.type ]
            self.query.append(('({}),\n'.format(
                ' ; '.join(
                    [ '{} = "{}"'.format(tlabel, x) for x in type_cons ]
                )
            ), 1))
        
        if node.label is not None and node.label not in self.labels: 
            self.labels.append(node.label)
            text, _, cost = S['/select_node.dl'](node)
            self.query.append((text, cost))
            if node.selects != [ 'gid' ]:
                self.set_output(
                    'source_text_{}'.format(node.label),
                    'start_line_{}'.format(node.label),
                    'start_col_{}'.format(node.label),
                    'end_line_{}'.format(node.label),
                    'end_col_{}'.format(node.label),
                    'gid_{}'.format(node.label)
                )
            else:
                self.set_output('gid_{}'.format(node.label))
        elif node.type is not None and node.label not in self.labels:
            text, _, cost = S['/node.dl'](node)
            self.query.append((text, cost))
        
        if node.type == '$literal':
            tlabel = 'type_{}'.format(node.id if node.label is None else node.label)
            self.query.append((str(
                Relation(self, "literal_types").args(tlabel)
            ) + ',\n', 2))

        if node.type == '$use':
            text, _, cost = S['/modifiers/use.dl'](node)
            self.query.append((text, cost)) 
        if node.type == "$df":
            text, outputs, cost = S['/modifiers/df.dl'](node)
            self.query.append((text, cost))
            self.set_output(outputs["to"])
            self.set_output(outputs["edge"])
        if node.has_mod(CBCallTarget):
            text, _, cost = S['/modifiers/call_target.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBModuleRoot):
            text, _, cost = S['/modifiers/module_root.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBAnyParentIs):
            text, _, cost = S['/modifiers/any_parent.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBAnyChildIs):
            text, _, cost = S['/modifiers/any_child.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBParentIs):
            text, _, cost = S['/modifiers/parent.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBRhs):
            text, _, cost = S['/modifiers/the_rhs.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBLhs):
            text, _, cost = S['/modifiers/the_lhs.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBObjectIs):
            text, _, cost = S['/modifiers/the_object.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBBodyIs):
            text, _, cost = S['/modifiers/the_body.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBNameIs):
            text, _, cost = S['/modifiers/the_name.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBOnlyLambdaParamIs):
            text, _, cost = S['/modifiers/only_lambda_param.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBFunctionIs):
            text, _, cost = S['/modifiers/the_function.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBAttributeIs):
            text, _, cost = S['/modifiers/the_attribute.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBSecondSubscriptIs):
            text, _, cost = S['/modifiers/second_subscript.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBFirstSubscriptIs):
            text, _, cost = S['/modifiers/first_subscript.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBSubscriptIs):
            text, _, cost = S['/modifiers/the_subscript.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBOnlySubscriptIs):
            text, _, cost = S['/modifiers/the_only_subscript.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBValueIs):
            text, _, cost = S['/modifiers/the_value.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBChildIs):
            text, _, cost = S['/modifiers/child.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBFirstChildIs):
            text, _, cost = S['/modifiers/first_child.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBSecondChildIs):
            text, _, cost = S['/modifiers/second_child.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBThirdChildIs):
            text, _, cost = S['/modifiers/third_child.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBNoThirdChild):
            text, _, cost = S['/modifiers/no_third_child.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBFirstArgIs):
            text, _, cost = S['/modifiers/first_arg.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBSecondArgIs):
            text, _, cost = S['/modifiers/second_arg.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBAnyArgIs):
            text, _, cost = S['/modifiers/any_arg.dl'](node)
            self.query.append((text, cost))
        
        # If we have a constraint on text, add that
        con = node.get_con(CBText)
        if con is not None:
            text, outputs, cost = S['/constraints/text.dl'](node)
            self.query.append((text, cost))
            self.query.append((str(
                Relation(self, "$eq").args(outputs['text'], '"{}"'.format(con.value))
            ) + ',\n', 1))
            self.pre_filter_words.append(con.value)

        # If we have the same text as some other part of the query,
        # enforce that constraint
        con = node.get_con(CBSameText)
        if con is not None:
            text, outputs, cost = S['/constraints/text.dl'](node)
            self.query.append((text, cost))
            self.query.append((str(
                Relation(self, "$eq").args(outputs['text'], 'source_text_{}'.format(con.value))
            ) + ',\n', 1))
        
        # If we have a constraint on name, add that 
        con = node.get_con(CBName)
        if con is not None:
            if name_var is None:
                text, outputs, cost = S['/constraints/name.dl'](node)
                name_var = outputs['name']
                self.query.append((text, cost))
            self.query.append((str(
                Relation(self, '$eq').args(name_var, '"{}"'.format(con.value))
            ) + ',\n', 1))
            self.pre_filter_words.append(con.value)

        # If we have a constraint on ids, add that 
        con = node.get_con(CBFromSet)
        if con is not None:
            con.write_file(self.inid)
            glabel = 'fid, gid_{}'.format(node.id if node.label is None else node.label)
            self.inputs += ".decl input_" + str(self.inid) + '(fid:Fid, gid:Gid)\n' + \
                ".input input_" + str(self.inid) + \
                '(IO=file, filename="{}", delimiter="\\t")\n\n'.format(con.file)
            self.query.append((str(
                Relation(self, 'input_' + str(self.inid)).args(glabel)
            ) + ',\n', len(con.frame) / 1000))
            self.inid += 1
            # Maybe constrain to the files that had these GIDs
            if con.files_constraint is not None:
                self.files_prefilters.append(con.files_constraint)

        # If we have a constraint on child count, add that 
        con = node.get_con(CBExactlyTwoChildren)
        if con is not None:
            text, _, cost = S['/constraints/exactly_two_children.dl'](node)
            self.query.append((text, cost))

        # If we have a constraint on normal arg count, add that 
        con = node.get_con(CBExactlyTwoNormalArgs)
        if con is not None:
            text, _, cost = S['/constraints/exactly_two_normal_args.dl'](node)
            self.query.append((text, cost))

        # If we have a constraint on child type, add that 
        con = node.get_con(CBEveryChildHasType)
        if con is not None:
            text, outputs, cost = S['/constraints/every_child_has_type.dl'](node)
            child_type_var = outputs['child_type']
            self.query.append((text, cost))
            tlabel = 'type_{}'.format(node.id if node.label is None else node.label)
            self.query.append((str(
                Relation(self, '$eq').args(child_type_var, '"{}"'.format(con.value))
            ) + ',\n', 1))


