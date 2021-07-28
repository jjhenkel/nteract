from .constraints import *
from .modifiers import *
from .infixer import Infixer
from .querybuilder import CB, CBSelect
from .evaluator import Evaluator

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
    def source_list_to_py_list(df, col):
        def _to_py_list(x):
            try:
                clean = str(x).replace('\\\n', '') # Line continuation
                clean = clean.replace('\\n', '').replace('\\t', ' ')
                return eval(clean)
            except:
                return ["???"]

        return ('[' + df[col].str.strip('[]') + ']').apply(
            _to_py_list
        )

    @staticmethod
    def get_comp_op(df, filter, lhs, rhs):
        def _get_comp_op(x):
            cleaned = str(x[0]).replace(str(x[1]), '').replace(str(x[2]), '').strip()
            
            if cleaned in ['!=', '==', '>', '<', '>=', '<=', 'is', 'is not', 'in', 'not in', '<>']:
                return cleaned
            return '<unk>'
        return df[[filter, lhs, rhs]].apply(_get_comp_op, axis=1)
  

###############################################################################
# SYNTAX NODES
# - usage type(... constraints ...)
###############################################################################

def imports(*args):
    return CB(['import_statement', 'import_from_statement']).with_constraints(*args)


def literal(*args):
    return CB('$literal').with_constraints(*args)


def comparison(*args):
    return CB('comparison_operator').with_constraints(*args)


def anything(*args):
    return CB('_').with_constraints(*args)


def attribute(*args):
    return CB('attribute').with_constraints(*args)


def param(*args):
    return CB('formal_parameter').with_constraints(*args)


def call(*args):
    return CB('call').with_constraints(*args)


def float_(*args):
    return CB('float').with_constraints(*args)


def keyword_argument(*args):
    return CB('keyword_argument').with_constraints(*args)


def string(*args):
    return CB('string').with_constraints(*args)


def identifier(*args):
    return CB('identifier').with_constraints(*args)


def integer(*args):
    return CB('integer').with_constraints(*args)


def for_loop(*args):
    return CB('for_statement').with_constraints(*args)


def while_loop(*args):
    return CB('while_statement').with_constraints(*args)


def function_def(*args):
    return CB('function_definition').with_constraints(*args)


def expression_stmt(*args):
    return CB('expression_statement').with_constraints(*args)


def slice_(*args):
    return CB('slice').with_constraints(*args)


def subscript(*args):
    return CB('subscript').with_constraints(*args)


def class_def(*args):
    return CB('class_definition').with_constraints(*args)


def comment(*args):
    return CB('comment').with_constraints(*args)


def assignment(*args):
    return CB('assignment').with_constraints(*args)


def list_(*args):
    return CB('list').with_constraints(*args)


def if_statement(*args):
    return CB('if_statement').with_constraints(*args)


def try_statement(*args):
    return CB('try_statement').with_constraints(*args)


def while_statement(*args):
    return CB('while_statement').with_constraints(*args)


def lambda_expression(*args):
    return CB('lambda').with_constraints(*args)


def import_stmt(*args):
    return CB('import_statement').with_constraints(*args)


def import_from_stmt(*args):
    return CB('import_from_statement').with_constraints(*args)


def dotted_name(*args):
    return CB('dotted_name').with_constraints(*args)


def wildcard_import(*args):
    return CB('wildcard_import').with_constraints(*args)

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


def module_name():
    return CB().with_mods(CBModuleName())


def imported_name():
    return CB().with_mods(CBImportedName())


def the_first_subscript():
    return CB().with_mods(CBFirstSubscriptIs())


def the_second_subscript():
    return CB().with_mods(CBSecondSubscriptIs())


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


def the_type():
    return CB().with_mods(CBTypeIs())


def uses():
    return CB().with_mods(CBUses())


def the_rhs():
    return CB().with_mods(CBRhs())


def the_lhs():
    return CB().with_mods(CBLhs())


def the_module_root():
    return CB().with_mods(CBModuleRoot())


def the_object():
    return CB().with_mods(CBObjectIs())


def the_body():
    return CB().with_mods(CBBodyIs())


def the_attribute():
    return CB().with_mods(CBAttributeIs())


def the_subscript():
    return CB().with_mods(CBSubscriptIs())


def the_only_subscript():
    return CB().with_mods(CBOnlySubscriptIs())


def the_value():
    return CB().with_mods(CBValueIs())


def the_name():
    return CB().with_mods(CBNameIs())


def the_function():
    return CB().with_mods(CBFunctionIs())


def call_target():
    return CB().with_mods(CBCallTarget())


def every_child():
    return CB().with_mods(CBEveryChildIs())


def the_only_lambda_param():
    return CB().with_mods(CBOnlyLambdaParamIs())


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


def the_value_is(subq):
    return the_value().merge(subq)


def the_body_is(subq):
    return the_body().merge(subq)


def the_subscript_is(subq):
    return the_subscript().merge(subq)


def the_first_subscript_is(subq):
    return the_first_subscript().merge(subq)


def the_second_subscript_is(subq):
    return the_second_subscript().merge(subq)


def the_first_arg_is(subq):
    return the_first_arg().merge(subq)


def the_second_arg_is(subq):
    return the_second_arg().merge(subq)


def the_only_subscript_is(subq):
    return the_only_subscript().merge(subq)


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


def the_object_is(subq):
    return the_object().merge(subq)


def call_target_is(subq):
    return call_target().merge(subq)


def the_module_root_is(subq):
    return the_module_root().merge(subq)


def any_arg_is(subq):
    return any_arg().merge(subq)


def module_name_is(subq):
    return module_name().merge(subq)


def imported_name_is(subq):
    return imported_name().merge(subq)


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


def the_only_lambda_param_is(subq):
    return the_only_lambda_param().merge(subq)


def the_function_is(subq):
    return the_function().merge(subq)


def the_name_is(subq):
    return the_name().merge(subq)


def the_type_is(subq):
    return the_type().merge(subq)

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
def execute(x, compile=False, debug=False, prefilter_files=None):
    return Evaluator(
        x, should_debug=debug, prefilter_files=prefilter_files
    ).eval(compile=compile)

def execute_union(*qs, compile=False, prefilter_files=None):
    return pd.concat(
        Evaluator(x).eval(
            compile=compile, prefilter_files=prefilter_files
        ) for x in qs 
    )

def visualize(frame, focus):
    from IPython import display

    data = frame.copy()

    reformated = {}
    data['fpath'] = data['fid'].apply(
        lambda x: Evaluator.resolve_fid(x)
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
        'application/code-book-matches+json': { 'results': [reformated], 'lang': 'python' }
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


class Queries:
    @staticmethod
    def qualified_call(i,c): 
        return execute(
            call(with_name(c)) % 'call'
            |where| call_target()
                |isa| use_of(imports(with_name(i))),
            compile=True
        ).assign(call_type=c, import_type=i) 

    @staticmethod
    def qualified_calls(i, cs): 
        temp1 = execute(
            imports(with_name(i)) % 'import',
            compile=True
        )

        temp2 = execute(
            use_of(imports(from_set(temp1, 'gid_import'))) % 'use',
            compile=True
        )

        qc = lambda c: execute(
            call(with_name(c)) % 'call'
            |where| call_target() |is_| anything(from_set(temp2, 'gid_use')),
            compile=True
        ).assign(call_type=c, import_type=i) 

        return pd.concat([
           qc(c) for c in cs
        ])


