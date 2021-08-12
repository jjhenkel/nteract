from .constraints import *
from .modifiers import *
from .infixer import Infixer
from .querybuilder import CB, CBSelect
from .evaluator import Evaluator

import json
import pandas as pd


class Utils:
    @staticmethod
    def get_literal_type(df, col):
        def _get_type(x):
            try:
                return eval('type(' + x + ').__name__')
            except:
                return 'unknown'
        return df[col].apply(_get_type).str.lower()
    
    @staticmethod
    def source_list_to_py_list(df, col, use_dask=False):
        def _to_py_list(x):
            try:
                clean = str(x).replace('\\\\n', '') # Line continuation
                clean = clean.replace('\\\n', '')
                clean = clean.replace('\\n', '').replace('\\t', ' ')
                return json.dumps(eval(clean))
            except:
                return json.dumps(["???"])

        if use_dask:
            return ('[' + df[col].str.strip('[]') + ']').apply(
                _to_py_list, meta='object'
            )
        else:
            return ('[' + df[col].str.strip('[]') + ']').apply(
                _to_py_list
            )
  

###############################################################################
# SYNTAX NODES
# - usage type(... constraints ...)
###############################################################################

def literal(*args):
    return CB('$literal').with_constraints(*args)


def member_expression(*args):
    return CB('member_access_expression').with_constraints(*args)


def anything(*args):
    return CB('_').with_constraints(*args)


def call(*args):
    return CB('invocation_expression').with_constraints(*args)


def string(*args):
    return CB('string_literal').with_constraints(*args)


def identifier(*args):
    return CB('identifier').with_constraints(*args)



###############################################################################
# CONSTRAINTS
# - usage type(... constraints ...)
###############################################################################


def with_name(name):
    return CBName(name)


def with_text(text):
    return CBText(text)


def with_exactly_two_children():
    return CBExactlyTwoChildren()


def where_every_child_has_type(the_type):
    return CBEveryChildHasType(the_type)


def same_text_as(target):
    return CBSameText(target)


def from_set_old(series):
    return CBFromSet(set(series.values))


def from_set(frame, col):
    inframe = pd.DataFrame()
    inframe['fid'] = frame['fid']
    inframe['gid'] = frame[col]
    return CBFromSet(inframe, files=set(frame.fid.unique()))


###############################################################################
# MODIFIERS (Operator Versions)
##  - usage |where| operator()
# |isa| subquery()
###############################################################################


def any_parent():
    return CB().with_mods(CBAnyParentIs(), CBRefTo())


def any_arg():
    return CB().with_mods(CBAnyArgIs(), CBRefTo())


def any_child():
    return CB().with_mods(CBAnyChildIs(), CBRefTo())


def the_first_arg():
    return CB().with_mods(CBFirstArgIs())


def the_second_arg():
    return CB().with_mods(CBSecondArgIs())


def the_first_child():
    return CB().with_mods(CBFirstChildIs())


def the_second_child():
    return CB().with_mods(CBSecondChildIs())


def the_third_child():
    return CB().with_mods(CBThirdChildIs())


def the_first_arg():
    return CB().with_mods(CBFirstArgIs())


def the_second_arg():
    return CB().with_mods(CBSecondArgIs())


def uses():
    return CB().with_mods(CBUses())


def the_rhs():
    return CB().with_mods(CBRhs())


def the_lhs():
    return CB().with_mods(CBLhs())


def the_body():
    return CB().with_mods(CBBodyIs())


def the_attribute():
    return CB().with_mods(CBAttributeIs())


def the_member():
    return CB().with_mods(CBMemberIs())


def call_target():
    return CB().with_mods(CBCallTarget())


def every_child():
    return CB().with_mods(CBEveryChildIs())


def the_expression():
    return CB().with_mods(CBExpressionIs())


def the_function():
    return CB().with_mods(CBFunctionIs())


def child():
    return CB().with_mods(CBChildIs())


def parent():
    return CB().with_mods(CBParentIs())


def no_third_child():
    return CB().with_mods(CBNoThirdChild())

###############################################################################
# MODIFIERS (Subquery Versions)
# - usage |where| operator(
# subquery() ...
# )
###############################################################################

def uses_are(subq):
    return uses().merge(subq)


def the_body_is(subq):
    return the_body().merge(subq)


def the_function_is(subq):
    return the_function().merge(subq)


def the_first_arg_is(subq):
    return the_first_arg().merge(subq)


def the_second_arg_is(subq):
    return the_second_arg().merge(subq)


def child_is(subq):
    return child().merge(subq)


def parent_is(subq):
    return parent().merge(subq)


def every_child_is(subq):
    return every_child().merge(subq)


def the_rhs_is(subq):
    return the_rhs().merge(subq)


def the_lhs_is(subq):
    return the_lhs().merge(subq)


def the_attribute_is(subq):
    return the_attribute().merge(subq)


def call_target_is(subq):
    return call_target().merge(subq)


def any_arg_is(subq):
    return any_arg().merge(subq)


def any_parent_is(subq):
    return any_parent().merge(subq)


def any_child_is(subq):
    return any_child().merge(subq)


def the_first_arg_is(subq):
    return the_first_arg().merge(subq)


def the_first_child_is(subq):
    return the_first_child().merge(subq)


def the_second_child_is(subq):
    return the_second_child().merge(subq)


def the_third_child_is(subq):
    return the_third_child().merge(subq)


def the_second_arg_is(subq):
    return the_second_arg().merge(subq)


def the_expression_is(subq):
    return the_expression().merge(subq)


def the_member_is(subq):
    return the_member().merge(subq)


###############################################################################
# (NAMED) INFIX OPERATORS
# - usage A |operator| B
###############################################################################


where = Infixer(
    lambda x, y: x.nest(y)
)

and_w = Infixer(
    lambda x, y: x.bubbleup().nest(y)
)

isa = Infixer(
    lambda x, y: x.merge(y)
)
is_ = isa # alias

use_of = lambda x: CB('$use').nest(x)
data_flows = lambda source, sink: CB('$df').nest(source).bubbleup().nest(sink)

###############################################################################

count = lambda: '$count'
select = lambda *s: CBSelect(*s)
select_as = lambda x,y: CBSelect(x, asname=y)

# TODO: fornow
def execute(x, compile=False, debug=False, use_dask=False, limit=None, prefilter_files=None, partition=None):
    return Evaluator(
        x, should_debug=debug, prefilter_files=prefilter_files
    ).eval(compile=compile, use_dask=use_dask, limit=limit, partition=partition)

def visualize(frame, focus):
    from IPython import display

    import glob
    import xxhash
    import hashlib
    import os

    if Evaluator._fids_to_fpaths is None:
        print("Loading visualization metadata; this may take a minute...")
        fpaths = []
        for chunk in glob.glob('{}/chunked/chunk-*/listing*'.format(Evaluator._dataset)):
            with open(chunk) as fh:
                fpaths.extend([ x.strip() for x in fh.readlines() if len(x.strip()) > 0 ])

        def hashit(thing):
            return int.from_bytes(
                xxhash.xxh64(thing, seed=3235823838).intdigest().to_bytes(8, byteorder='little'),
                signed=True, byteorder="little"
            )
        
        def sha256sum(filename):
            h  = hashlib.sha256()
            b  = bytearray(128*1024)
            mv = memoryview(b)
            with open(filename, 'rb', buffering=0) as f:
                for n in iter(lambda : f.readinto(mv), 0):
                    h.update(mv[:n])
            return h.hexdigest()
                
        fids_to_fpaths = {}
        for fpath in fpaths:
            vfpath = sha256sum(fpath.replace('data/raw', 'data/csharp/raw')) + '.cs'
            fids_to_fpaths[hashit('parsed/' + vfpath + '.xml.gz')] = fpath

        Evaluator._fids_to_fpaths = fids_to_fpaths
        print("  + Loaded!")

    data = frame.copy()

    reformated = {}
    data['fpath'] = data['fid'].apply(
        lambda x: Evaluator.resolve_fid(x).replace('data/raw', 'cs-data') # TODO: hack
    )

    keys = []
    sorts = ['fpath']
    for col in data.columns:
        if col.startswith('start_line_'):
            keys.append(col.replace('start_line_', ''))
            reformated[keys[-1] if keys[-1] != focus else '$match'] = []
            if keys[-1] == focus:
                sorts.append(col)
                sorts.append(col.replace('start_line_', 'start_col_'))
    
    data_sorted = data.sort_values(sorts)

    for key in keys:
        for _, row in data_sorted.iterrows():
            reformated[key if key != focus else '$match'].append({
                's_line': row['start_line_' + key],
                's_col': row['start_col_' + key],
                'e_line': row['end_line_' + key],
                'e_col': row['end_col_' + key],
                'text': row['source_text_' + key],
                'fpath': row['fpath'],
                'gid': row['gid_' + key]
            })

    display.display({
        'application/code-book-matches+json': { 'results': [reformated], 'lang': 'csharp' }
    }, raw=True)



def show_dot(dot):
    from IPython import display

    display.display({
        'application/code-book-matches+json': { 'dot': dot, 'lang': 'graphviz' }
    }, raw=True)


def showgid(gid):
    visualize(execute(anything(from_set(pd.Series([gid, -gid]))) % 'targ'), 'targ')


###############################################################################
# MISC / PRE-Made queries
###############################################################################


def danger_reset_paths():
    import os
    import glob
    import shutil
    
    for outdir in glob.glob('/tmp/query-outputs/*'):
        shutil.rmtree(outdir)
    for indir in glob.glob('/tmp/query-inputs/*'):
        shutil.rmtree(indir)
    if os.path.exists('/tmp/queries'):
        shutil.rmtree('/tmp/queries')
    if os.path.exists('/tmp/query-stats'):
        shutil.rmtree('/tmp/query-stats')
    os.mkdir('/tmp/queries')    
    os.mkdir('/tmp/query-stats')    

