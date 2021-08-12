# Need to help python find our version of arrow
import sys
sys.path.append('/arrow/python')
###############################################

import os
import io
import json
import tqdm
import glob
import time
import xxhash
import secrets
import hashlib
import subprocess
import pandas as pd
import dask.dataframe as ddf
import pyarrow.parquet as pq
import pyarrow as pa
from .constraints import *
from .modifiers import *
from .querybuilder import CB
from .pathman import Pathman
from .snippets import SNIPPETS as S, refresh_snippet_offsets


def chunks(lst, n):
    """Yield n successive (roughly even) chunks from lst."""
    n = len(lst) // n
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# NOTE: Not sure if this makes sense/is any good
# on each fresh import we should be getting a fresh
# session id
SESSION_ID = secrets.token_hex(16)


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
        return ", ".join([self.process_arg(x) for x in self.arguments])

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
        xxhash.xxh64(thing, seed=3235823838).intdigest().to_bytes(
            8, byteorder='little'),
        signed=True, byteorder="little"
    )


class Evaluator:
    _dataset = '/data/csharp'
    _fids_to_fpaths = None

    @staticmethod
    def resolve_fid(fid):
        return Evaluator._fids_to_fpaths[fid]

    def __init__(self, query, should_debug=False, prefilter_files=None):
        query.idify()

        self.inid = 1

        self.worklist = [query]
        self.debug = should_debug

        self.outputs = []
        self.outputs_typed = []
        self.query = []
        self.inputs = ''
        self.dataset = Evaluator._dataset
        self.session_id = SESSION_ID
        self.pathman = Pathman(self.dataset, self.session_id)
        self.labels = []
        self.delayed_inputs = []

        self.pre_filter_words = []

        self.files_prefilters = []

        if prefilter_files is not None:
            self.files_prefilters.append(set(prefilter_files))

        self.set_output("fid")

    def build_query(self, compiled):
        # Traverse the tree / visit each node / build the query
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

        # Get query components (header/body/footer)
        header, body, footer = self.get_header(
            compiled), self.get_body(), self.get_footer()

        if self.debug:
            # Maybe just dump query and quit
            print(header + body + footer)
            return False

        # This is so if we update prelude.dl/utils.dl we invalidate
        # query hashes
        extra_to_hash = self.pathman.get_prelude_file() + self.pathman.get_utils_file()

        # Get the hash of this query and inform pathman of the new
        # query hash!
        query_hash = hashlib.sha256(
            (header + body + footer + extra_to_hash).encode('utf-8')
        ).hexdigest()
        self.pathman.set_query_hash(query_hash)
        
        # Now handle the input files
        for con, in_id in self.delayed_inputs:
            con.write_file(self.pathman.get_input_file_path(in_id))
            self.inputs += ".decl input_" + str(in_id) + '(fid:Fid, gid:Gid)\n' + \
                ".input input_" + str(in_id) + \
                '(IO=file, filename="{}", delimiter="\\t")\n\n'.format(con.file)

        # And now RESET the query hash --- this was tricky
        header, body, footer = self.get_header(
            compiled), self.get_body(), self.get_footer()

        # Get the hash of this query and inform pathman of the new
        # query hash!
        query_hash = hashlib.sha256(
            (header + body + footer + extra_to_hash).encode('utf-8')
        ).hexdigest()
        self.pathman.set_query_hash(query_hash)

        # Write the actual query
        self.pathman.write_query_dl(self.get_header(compiled) + body + footer)

        # Write a query that's suitable to pass to an interactive
        # version of souffle so we can get profiling data (if necessary)
        # for -PSIPS:profile-use (which re-orders relations)
        self.pathman.write_query_prof(self.get_header(False) + body + footer)

        return True

    def compile_query(self):
        if self.pathman.get_query_is_compiled():
            print(
                '  + Query already compiled (cached) `{}`'.format(self.pathman.get_query_hash()))
            return True

        # Pre-flight / profiling (profile step)
        start = time.perf_counter()

        profile_result = subprocess.run([
            '/usr/bin/time',
            '-vo',
            self.pathman.get_query_stats_file_path('.profile'),
            self.pathman.get_real_souffle_path(),
            '-S',                   # Stratify
            '--profile-frequency',  # Profile it all
            '-p', self.pathman.get_query_prof_file_path(),
            '-F', self.pathman.get_profiler_chunk(),
            self.pathman.get_query_prof_dl_file_path()
        ], capture_output=True)

        if profile_result.stderr != b'' and b'error' in profile_result.stderr:
            print("Profiling error (profile step)")
            print(profile_result.stderr)
            return False

        elapsed_time = time.perf_counter() - start
        print(f"  + Profile time: {elapsed_time:.4f}s")

        # Actually compile (generate step)
        start = time.perf_counter()
        compile_result = subprocess.run([
            '/usr/bin/time',
            '-vo',
            self.pathman.get_query_stats_file_path('.generate'),
            self.pathman.get_real_souffle_path(),
            '-S',                 # Stratify
            '-PSIPS:profile-use',  # Re-order based on profiling data
            '-u', self.pathman.get_query_prof_file_path(),
            '-g-', self.pathman.get_query_dl_file_path()
        ], capture_output=True)

        if compile_result.stderr != b'' and b'error' in compile_result.stderr:
            print("Souffle compile error (generate step)")
            print(compile_result.stderr)
            return False

        # We read this / write it back out because, previously, we had
        # a certain patch we applied to the raw cpp file
        the_program = compile_result.stdout.decode('utf-8')
        self.pathman.write_query_cpp(the_program)

        # Actually compile (cpp -> bin step)
        compile_result = subprocess.run([
            '/usr/bin/time',
            '-vo',
            self.pathman.get_query_stats_file_path('.compile'),
            self.pathman.get_souffle_compiler_path(),
            self.pathman.get_query_cpp_file_path()
        ], capture_output=True)
        elapsed_time = time.perf_counter() - start

        if compile_result.stderr != b'' and b'error' in compile_result.stderr:
            print("Souffle compile error (cpp->bin step)")
            print(compile_result.stderr)
            return False

        # This lets us use cached versions of queries later
        self.pathman.move_query_files_to_global_cache()

        print(f"  + Compile time: {elapsed_time:.4f}s")
        return True

    def eval(self, compile=False, use_dask=False, limit=None, partition=None):
        if limit is not None:
            assert partition is None, "Can't use limit with partition."
        if partition is not None:
            assert isinstance(partition, tuple) and len(partition) == 2, "Partition must be (x, N) tuple."

        refresh_snippet_offsets()

        total_start = time.perf_counter()

        # Build the query first
        if not self.build_query(compile):
            return pd.DataFrame()

        # Maybe compile/check status
        if compile and not self.compile_query():
            print('  ! Query compilation failed')
            return

        # self.select_files()
        # TODO: return to reverse-index for file select

        targets = self.pathman.get_target_chunks(limit=limit)

        if partition is not None:
            print('  + Selecting partition {}/{}'.format(partition[0] + 1, partition[1]))
            targets = list(chunks(targets, partition[1]))[partition[0]]

        start = time.perf_counter()

        # Make the command
        full_cmd = '/usr/bin/time -vo "{}" parallel -k '.format(
            self.pathman.get_query_stats_file_path('.execution')
        )
        cmd_inputs = (
            '\n'.join(targets) + '\n'
        ).encode('utf-8')
        if compile:
            # Compiled mode. Use wrapper/bin file
            full_cmd += self.pathman.get_wrapper_path()
            full_cmd += " "
            full_cmd += self.pathman.get_query_bin_file_path(nosession=True)
            full_cmd += " -F{} -D"
            full_cmd += self.pathman.get_output_path_prefix()
        else:
            # Interpreted mode. Use souffle/dl file
            full_cmd += self.pathman.get_souffle_path()
            full_cmd += " -F{} -D"
            full_cmd += self.pathman.get_output_path_prefix()
            full_cmd += " "
            full_cmd += self.pathman.get_query_dl_file_path()

        exec_result = subprocess.run([
            'bash', '-c', full_cmd
        ], input=cmd_inputs, capture_output=True)
        elapsed_time = time.perf_counter() - start
        print(f"  + Query time: {elapsed_time:.4f}s")

        start = time.perf_counter()

        final = None
        if use_dask:
            final = ddf.read_parquet(
                self.pathman.get_output_files(targets)
            )
        else:
            final = pq.ParquetDataset(
                self.pathman.get_output_files(targets)
            ).read().to_pandas()

        elapsed_time = time.perf_counter() - start
        print(f"  + Collation time: {elapsed_time:.4f}s")

        elapsed_time = time.perf_counter() - total_start
        print(f"Total time: {elapsed_time:.4f}s")

        return final

    def get_body(self):
        clauses = sorted(self.query, key=lambda x: x[1])
        query = ''.join([x[0] for x in clauses])
        return query.strip().rstrip(',') + '\n  .\n'

    def get_footer(self):
        footer = ".output synth_query(IO=parquet, filename=\"results\")\n"
        return footer

    def get_header(self, compiled):
        final = ""
        final += '#include "{}"\n'.format(
            self.pathman.get_prelude_file_path()
        )
        final += self.inputs
        final += ".decl synth_query(" + ", ".join(self.outputs_typed) + ")\n"
        final += "synth_query(" + ", ".join(self.outputs) + ") :- \n"
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
            elif arg.startswith('out_name') or arg.startswith('out_def_name'):
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
            tlabel = 'type_{}'.format(
                node.id if node.label is None else node.label)
            type_cons = node.type if isinstance(
                node.type, list) else [node.type]
            self.query.append(('({}),\n'.format(
                ' ; '.join(
                    ['{} = "{}"'.format(tlabel, x) for x in type_cons]
                )
            ), 1))

        if node.label is not None and node.label not in self.labels:
            self.labels.append(node.label)
            text, _, cost = S['/select_node.dl'](node)
            self.query.append((text, cost))
            if node.selects != ['gid']:
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
            tlabel = 'type_{}'.format(
                node.id if node.label is None else node.label)
            self.query.append((str(
                Relation(self, "literal_types").args(tlabel)
            ) + ',\n', 2))

        if node.type == '$use':
            text, outputs, cost = S['/modifiers/use.dl'](node)
            if 'def_name' in node.selects:
                if node.select_as:
                    self.query.append((str(
                        Relation(self, '$eq').args(
                            node.select_as, outputs['def_name'])
                    ) + ',\n', 1))
                    self.set_output(node.select_as)
                else:
                    self.set_output(
                        outputs['def_name']
                    )
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
        if node.has_mod(CBMemberIs):
            text, _, cost = S['/modifiers/the_member.dl'](node)
            self.query.append((text, cost))
        if node.has_mod(CBExpressionIs):
            text, _, cost = S['/modifiers/the_expression.dl'](node)
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
                Relation(self, "$eq").args(
                    outputs['text'], '"{}"'.format(con.value))
            ) + ',\n', 1))
            self.pre_filter_words.append(con.value)

        # If we have the same text as some other part of the query,
        # enforce that constraint
        con = node.get_con(CBSameText)
        if con is not None:
            text, outputs, cost = S['/constraints/text.dl'](node)
            self.query.append((text, cost))
            self.query.append((str(
                Relation(self, "$eq").args(
                    outputs['text'], 'source_text_{}'.format(con.value))
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
            self.delayed_inputs.append((
                con, self.inid
            ))
            glabel = 'fid, gid_{}'.format(
                node.id if node.label is None else node.label)
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
            text, outputs, cost = S['/constraints/every_child_has_type.dl'](
                node)
            child_type_var = outputs['child_type']
            self.query.append((text, cost))
            tlabel = 'type_{}'.format(
                node.id if node.label is None else node.label)
            self.query.append((str(
                Relation(self, '$eq').args(
                    child_type_var, '"{}"'.format(con.value))
            ) + ',\n', 1))
