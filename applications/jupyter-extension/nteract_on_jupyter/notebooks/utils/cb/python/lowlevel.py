import json
import regex
import pickle
import os.path
import xxhash
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds 
import pyarrow.parquet as pq 
import pyarrow.gandiva as gd

TYPE_TO_BYTES = [
    'aliased_import',
    'argument_list',
    'assert_statement',
    'assignment',
    'attribute',
    'augmented_assignment',
    'await',
    'binary_operator',
    'block',
    'boolean_operator',
    'break_statement',
    'call',
    'chevron',
    'class_definition',
    'comment',
    'comparison_operator',
    'compound_statement',
    'concatenated_string',
    'conditional_expression',
    'continue_statement',
    'decorated_definition',
    'decorator',
    'default_parameter',
    'delete_statement',
    'dictionary',
    'dictionary_comprehension',
    'dictionary_splat',
    'dictionary_splat_pattern',
    'dotted_name',
    'elif_clause',
    'ellipsis',
    'else_clause',
    'escape_sequence',
    'except_clause',
    'exec_statement',
    'expression',
    'expression_list',
    'expression_statement',
    'f_alias',
    'f_alternative',
    'f_argument',
    'f_arguments',
    'f_attribute',
    'f_body',
    'f_cause',
    'f_child',
    'f_code',
    'f_condition',
    'f_consequence',
    'f_definition',
    'f_function',
    'f_key',
    'f_left',
    'f_module_name',
    'f_name',
    'f_object',
    'f_operator',
    'f_parameters',
    'f_return_type',
    'f_right',
    'f_subscript',
    'f_superclasses',
    'f_type',
    'f_value',
    'false',
    'finally_clause',
    'float',
    'for_in_clause',
    'for_statement',
    'format_expression',
    'format_specifier',
    'function_definition',
    'future_import_statement',
    'generator_expression',
    'global_statement',
    'identifier',
    'if_clause',
    'if_statement',
    'import_from_statement',
    'import_prefix',
    'import_statement',
    'integer',
    'interpolation',
    'keyword_argument',
    'lambda',
    'lambda_parameters',
    'list',
    'list_comprehension',
    'list_pattern',
    'list_splat',
    'list_splat_pattern',
    'module',
    'named_expression',
    'none',
    'nonlocal_statement',
    'not_operator',
    'pair',
    'parameter',
    'parameters',
    'parenthesized_expression',
    'parenthesized_list_splat',
    'pass_statement',
    'pattern',
    'pattern_list',
    'primary_expression',
    'print_statement',
    'raise_statement',
    'relative_import',
    'return_statement',
    'set',
    'set_comprehension',
    'simple_statement',
    'slice',
    'string',
    'subscript',
    'true',
    'try_statement',
    'tuple',
    'tuple_pattern',
    'type',
    'type_conversion',
    'typed_default_parameter',
    'typed_parameter',
    'unary_operator',
    'while_statement',
    'wildcard_import',
    'with_clause',
    'with_item',
    'with_statement',
    'yield',
]

__F_CACHE = {}
__DATA = None
__T_CACHE = {}
__M_CACHE = {}

def type_to_bytes(the_type):
    return TYPE_TO_BYTES.index(the_type).to_bytes(2, byteorder='little')

def type_to_idx(the_type):
    return TYPE_TO_BYTES.index(the_type)

QUERY_LANG = r"^((\{(>|<)?=?\d*\})(\^)?(\".*?\"=)?\(?([a-z_0-9]+)?\)?(\.f_[a-z_0-9]+)?(\[\d+\])?)+$"
SUB_QUERY_LANG = r"(\{(>|<)?=?\d*\})(\^)?(\".*?\"=)?\(?([a-z_0-9]+)?\)?(\.f_[a-z_0-9]+)?(\[\d+\])?"
QUERY_REGEX = regex.compile(QUERY_LANG)
SUB_QUERY_REGEX = regex.compile(SUB_QUERY_LANG)

def get_text_fragement(fid, start, end, debug=False):
    if debug:
        return 'get_file({}@{}:{})'.format(fid, start, end), 0, 0, 0, 0
    try:
        if fid not in __F_CACHE:
            with open('/data/raw-files/{}.txt'.format(fid), 'rb') as fh:
                __F_CACHE[fid] = fh.read().decode('utf-8')
        text = __F_CACHE[fid][start:end]
        last_newline = __F_CACHE[fid].rindex('\n', 0, start)
        start_col = start - last_newline
        start_line = __F_CACHE[fid].count('\n', 0, start)
        last_newline = __F_CACHE[fid].rindex('\n', 0, end)
        end_col = end - last_newline
        end_line = __F_CACHE[fid].count('\n', 0, end)
        return text, start_line + 1, start_col, end_line + 1, end_col
    except Exception as ex:
        return str(ex) + '\ncant find {}@{}:{}'.format(fid, start, end), 0, 0, 0, 0


def decode_op_dist(dist):
    if dist is None or len(dist.replace('{', '').replace('}', '')) <= 0:
        return 0, int(0)
    
    dist = dist[1:-1]
    if dist[0] == '=':
        return 5, int(dist[1:])
    elif dist[:2] == '<=':
        return 2, int(dist[2:])
    elif dist[:2] == '>=':
        return 4, int(dist[2:])
    elif dist[0] == '<':
        return 1, int(dist[1:])
    elif dist[0] == '>':
        return 3, int(dist[1:])
    else:
        return 0, int(dist)


def parse_query(query_string, type_to_idx, debug=False):
    builder = gd.TreeExprBuilder()

    if query_string[0] != '{':
        query_string = '{}' + query_string

    match = regex.match(QUERY_REGEX, query_string, version=regex.V1)

    target_func = 'match_tree_path_{}_'.format(len(match.captures(1)))
    generic_func = target_func + 'generic'

    params = []
    generic_params_pre = []
    generic_params_post = []

    first_label = '?'

    for i, sub_string in enumerate(match.captures(1)):
        sub_match = regex.match(SUB_QUERY_REGEX, sub_string, version=regex.V1)
        steps, _, negate, name, label, field, index = sub_match.groups()

        if first_label == '?':
            first_label = label

        negate = negate == '^'

        match_name = name is not None
        name = name[1:-2] if match_name else None
        
        match_label = label is not None
        label = type_to_idx(label) if match_label else 0

        match_field = field is not None
        field = type_to_idx(field[1:]) if match_field else 0

        match_index = index is not None
        index = int(index[1:-1]) if match_index else 0
    
        steps_op, steps_dist = decode_op_dist(steps)

        target_func += ('1' if negate else '0')
        generic_params_pre.append(builder.make_literal(negate, pa.bool_()))
        target_func += ('1' if match_label else '0')
        generic_params_pre.append(builder.make_literal(match_label, pa.bool_()))
        target_func += ('1' if match_name else '0')
        generic_params_pre.append(builder.make_literal(match_name, pa.bool_()))
        target_func += ('1' if match_field else '0')
        generic_params_pre.append(builder.make_literal(match_field, pa.bool_()))
        target_func += ('1' if match_index else '0')
        generic_params_pre.append(builder.make_literal(match_index, pa.bool_()))
        
        if match_label:
            params.append(builder.make_literal(label, pa.uint16()))
            generic_params_post.append(builder.make_literal(label, pa.uint16()))
        else:
            generic_params_post.append(builder.make_literal(0, pa.uint16()))


        if match_name:
            as_hash = int.from_bytes(
                xxhash.xxh64(name, seed=3235823838).intdigest().to_bytes(8, byteorder='little'),
                signed=True, byteorder="little"
            )
            params.append(builder.make_literal(as_hash, pa.int64()))
            generic_params_post.append(builder.make_literal(as_hash, pa.int64()))
        else:
            generic_params_post.append(builder.make_literal(0, pa.int64()))
        
        if match_field:
            params.append(builder.make_literal(field, pa.uint16()))
            generic_params_post.append(builder.make_literal(field, pa.uint16()))
        else:
            generic_params_post.append(builder.make_literal(0, pa.uint16()))

        if match_index:
            params.append(builder.make_literal(index, pa.uint16()))
            generic_params_post.append(builder.make_literal(index, pa.uint16()))
        else:
            generic_params_post.append(builder.make_literal(0, pa.uint16()))
            
        if steps_op == 5:
            target_func += '3'
            params.append(builder.make_literal(steps_dist, pa.uint16()))
            generic_params_post.append(builder.make_literal(steps_dist, pa.uint16()))
            generic_params_pre.append(builder.make_literal(3, pa.uint16()))
        elif steps_op == 4:
            target_func += '2'
            params.append(builder.make_literal(steps_dist - 1, pa.uint16()))
            generic_params_post.append(builder.make_literal(steps_dist - 1, pa.uint16()))
            generic_params_pre.append(builder.make_literal(2, pa.uint16()))
        elif steps_op == 3:
            target_func += '2'
            params.append(builder.make_literal(steps_dist, pa.uint16()))
            generic_params_post.append(builder.make_literal(steps_dist, pa.uint16()))
            generic_params_pre.append(builder.make_literal(2, pa.uint16()))
        elif steps_op == 2:
            target_func += '1'
            params.append(builder.make_literal(steps_dist + 1, pa.uint16()))
            generic_params_post.append(builder.make_literal(steps_dist + 1, pa.uint16()))
            generic_params_pre.append(builder.make_literal(1, pa.uint16()))
        elif steps_op == 1:
            target_func += '1'
            params.append(builder.make_literal(steps_dist, pa.uint16()))
            generic_params_post.append(builder.make_literal(steps_dist, pa.uint16()))
            generic_params_pre.append(builder.make_literal(1, pa.uint16()))
        else:
            target_func += '0'
            generic_params_post.append(builder.make_literal(0, pa.uint16()))
            generic_params_pre.append(builder.make_literal(0, pa.uint16()))

        target_func += '_'
    
    target_func = target_func[:-1]
    if debug:
        print(first_label, target_func, params)


    generic_params = (generic_params_pre, generic_params_post)
    return first_label, generic_func, target_func, generic_params, params, builder


def get_text_from_capture(res, cidx):
    offset = 32 + (cidx - 1) * 40
    return get_text_fragement(
        int.from_bytes(res[0:8], signed=True, byteorder="little"),
        int.from_bytes(res[offset+0:offset+4], signed=False, byteorder="little"),
        int.from_bytes(res[offset+4:offset+8], signed=False, byteorder="little")
    )

def get_texts_from_capture(res, cidx, node_type, limit=None):
    offset = 32 + (cidx - 1) * 40
    gid_offset = 8 + (cidx - 1) * 40
    out = []
    for r in res[:limit] if limit is not None else res:
        gid = str(int.from_bytes(r[gid_offset:gid_offset+8], signed=True, byteorder="little"))
        fid = str(int.from_bytes(r[0:8], signed=True, byteorder="little"))
        sidx = int.from_bytes(r[offset+0:offset+4], signed=False, byteorder="little")
        eidx = int.from_bytes(r[offset+4:offset+8], signed=False, byteorder="little")
        text, sl, sc, el, ec = get_text_fragement(fid, sidx, eidx)
        out.append({
            'gid': gid,
            'fid': fid,
            's_line': sl,
            's_col': sc,
            'e_line': el,
            'e_col': ec,
            'text': text,
            'type': node_type,
            'project': __M_CACHE[fid][0] if fid in __M_CACHE else '???',
            'version': __M_CACHE[fid][1] if fid in __M_CACHE else '???',
            'file_path': __M_CACHE[fid][2] if fid in __M_CACHE else '???',
        })
    return out


def query_python(query_string, extra="file_id", name_is=None, name_regex=None):
    global __DATA, __T_CACHE
    
    root_type, generic_func, target_func, generic_params, params, builder = parse_query(
        query_string, type_to_idx
    )

    as_table = None
    proj = None

    if __DATA is None:
        __DATA = ds.dataset('/data/parquet', format='parquet', partitioning='hive')

    if root_type not in __T_CACHE:
        the_filter = ds.field('type') == root_type

        extra_cols = [extra, 'gid', 'project', 'version', 'file_path']
        if name_is is not None:
            extra_cols.append('name')
            the_filter = the_filter & (ds.field('name') == name_is)
        elif name_regex is not None:
            print('Regex name filter not yet supported')

        __T_CACHE[root_type] = __DATA.to_table(
            columns=['path'] + extra_cols,
            filter=the_filter
        )
        
    
    as_table = __T_CACHE[root_type]
        
    params = [
        builder.make_field(as_table.schema.field(extra)),
        builder.make_field(as_table.schema.field('path'))
    ] + params
    generic_params = generic_params[0] + [
        builder.make_field(as_table.schema.field(extra)),
        builder.make_field(as_table.schema.field('path'))
    ] + generic_params[1]

    proj = None

    try:
        proj = gd.make_projector(as_table.schema, [
            builder.make_expression(
                builder.make_function(target_func, params, pa.binary()),
                pa.field("result", pa.binary())
            )
        ], pa.default_memory_pool())
    except:
        proj = gd.make_projector(as_table.schema, [
            builder.make_expression(
                builder.make_function(generic_func, generic_params, pa.binary()),
                pa.field("result", pa.binary())
            )
        ], pa.default_memory_pool())

    total = []
    for record_batch in as_table.to_batches():
        res, = proj.evaluate(record_batch)
        for i, row in res.to_pandas().items():
            if row == b'':
                continue
            total.append(row)
            if str(record_batch['file_id'][i]) not in __M_CACHE:
                __M_CACHE[str(record_batch['file_id'][i])] = (
                    str(record_batch['project'][i]),
                    str(record_batch['version'][i]),
                    str(record_batch['file_path'][i])
                )
    
    if len(total) > 0:
        return pd.Series(total)
    else:
        return pd.Series([ b'' ])


def merge_paths(series_l, series_r, on):
    on_l, on_r = on

    frame_l = series_l.copy()
    if not isinstance(series_l, pd.DataFrame):
        frame_l = series_l.copy().to_frame(name="dat")

    frame_r = series_r.copy()
    if not isinstance(series_r, pd.DataFrame):
        frame_r = series_r.copy().to_frame(name="dat")

    target_l = None
    if on_l.startswith('left.') or on_l.startswith('right.'):
        parts = on_l.split('.')[:-2][::-1]
        the_ref = 'dat_' + '_'.join([ part[0] for part in parts ])
        target_l = frame_l[the_ref]
        on_l = '.'.join(on_l.split('.')[-2:])
    else:
        target_l = frame_l.dat

    target_r = None
    if on_r.startswith('left.') or on_r.startswith('right.'):
        parts = on_r.split('.')[:-2][::-1]
        the_ref = 'dat_' + '_'.join([ part[0] for part in parts ])
        target_r = frame_r[the_ref]
        on_r = '.'.join(on_r.split('.')[-2:])
    else:
        target_r = frame_r.dat

    if on_l.startswith('defs.'):
        cindex = int(on_l.replace('defs.', '')) - 1
        frame_l['key'] = target_l.str[16+40*cindex:24+40*cindex]
    elif on_l.startswith('gids.'):
        cindex = int(on_l.replace('gids.', '')) - 1
        frame_l['key'] = target_l.str[8+40*cindex:16+40*cindex]

    if on_r.startswith('defs.'):
        cindex = int(on_r.replace('defs.', '')) - 1
        frame_r['key'] = target_r.str[16+40*cindex:24+40*cindex]
    elif on_r.startswith('gids.'):
        cindex = int(on_r.replace('gids.', '')) - 1
        frame_r['key'] = target_r.str[8+40*cindex:16+40*cindex]
    
    frame_l.columns = frame_l.columns.map(lambda x: str(x) + '_l')
    frame_r.columns = frame_r.columns.map(lambda x: str(x) + '_r')

    return frame_l.merge(
        frame_r,
        how="inner",
        left_on="key_l",
        right_on="key_r"
    )


def get_results(result_set, labels):
    if len(result_set) == 1 and 0 in result_set and result_set[0] == b'':
        return { }

    def _get_all_labels(cur):
        if isinstance(cur, list):
            res = []
            for i, l in enumerate(cur):
                if l[0] is not None:
                    res.append(('dat', i + 1, l)) 
            return res
        
        return list(map(
            lambda x: (x[0] + '_l', x[1], x[2]),
            _get_all_labels(cur['left'])
        )) + list(map(
            lambda x: (x[0] + '_r', x[1], x[2]),
            _get_all_labels(cur['right'])
        ))
    
    results_map = { }
    for path, idx, (label, node_type) in _get_all_labels(labels):
        if path == 'dat':
            results_map[label] = get_texts_from_capture(result_set, idx, node_type)
        else:
            results_map[label] = get_texts_from_capture(result_set[path], idx, node_type)
    
    return results_map


def display_results(results, limit=10, just_text=False):
    if not just_text:
        from IPython import display
        display.display({
            'application/code-book-matches+json': { 'results': results, 'lang': 'python' }
        }, raw=True)
        return

    for rs, results_map in enumerate(results):
        if '$match' not in results_map:
            continue
        other_keys = sorted([ k for k in results_map.keys() if k != '$match' ])
        for i, val in enumerate(results_map['$match']):
            if i > limit:
                print('Over {} matches. Stopping early.'.format(limit))
                break
            print('Match (RS#{}):\n```\n{}\n```'.format(
                rs + 1, val['text']
            ))
            for j, k in enumerate(other_keys):
                print(' ' * j + '└─ {}: ```\n{}{}\n{}```'.format(
                    k,
                    ' ' * (j+3),
                    results_map[k][i]['text'].replace('\n', '\n' + ' ' * (j+3)),
                    ' ' * (j+3)
                ))
        
