from .constraints import *
from .modifiers import *
from .infixer import Infixer
from .querybuilder import CB

###############################################################################
# SYNTAX NODES
# - usage type(... constraints ...)
###############################################################################


def new(*args):
    return CB('object_creation_expression').with_constraints(*args)


def param(*args):
    return CB('formal_parameter').with_constraints(*args)


def call(*args):
    return CB('method_invocation').with_constraints(*args)


def string(*args):
    return CB('string_literal').with_constraints(*args)


def integer(*args):
    return CB('decimal_integer_literal').with_constraints(*args)


def method_param(*args):
    return CB('formal_parameter').with_constraints(*args)


def annotation(*args):
    return CB('annotation').with_constraints(*args)


def method_declaration(*args):
    return CB('method_declaration').with_constraints(*args)


def field_ref(*args):
    return CB('field_declaration_ref').with_constraints(*args)


def type_(*args):
    return CB('type_identifier').with_constraints(*args)


def comment(*args):
    return CB('comment').with_constraints(*args)


def comment(*args):
    return CB('comment').with_constraints(*args)


def field_access(*args):
    return CB('field_access').with_constraints(*args)


def assert_statement(*args):
    return CB('assert_statement').with_constraints(*args)


def assignment_expression(*args):
    return CB('assignment_expression').with_constraints(*args)


def binary_expression(*args):
    return CB('binary_expression').with_constraints(*args)


def array_creation_expression(*args):
    return CB('array_creation_expression').with_constraints(*args)


def enhanced_for_statement(*args):
    return CB('enhanced_for_statement').with_constraints(*args)


def instanceof_expression(*args):
    return CB('instanceof_expression').with_constraints(*args)


def if_statement(*args):
    return CB('if_statement').with_constraints(*args)


def try_statement(*args):
    return CB('try_statement').with_constraints(*args)


def throw_statement(*args):
    return CB('throw_statement').with_constraints(*args)


def while_statement(*args):
    return CB('while_statement').with_constraints(*args)


def lambda_expression(*args):
    return CB('lambda_expression').with_constraints(*args)


def labeled_statement(*args):
    return CB('labeled_statement').with_constraints(*args)

###############################################################################
# CONSTRAINTS
# - usage type(... constraints ...)
###############################################################################


def with_name(name):
    return CBName(name)


def with_text(text):
    return CBText(text)


###############################################################################
# MODIFIERS (Operator Versions)
##  - usage |where| operator()
# |isa| subquery()
###############################################################################


def any_arg():
    return CB().with_mods(CBAnyArgIs(), CBRefTo())


def the_receiver():
    return CB().with_mods(CBReceiverIs(), CBRefTo())


def the_first_args_receiver():
    return call().with_mods(CBFirstArgIs(), CBRefTo()).nest(
        CB().with_mods(CBReceiverIs(), CBRefTo())
    )


def the_first_arg():
    return CB().with_mods(CBFirstArgIs(), CBRefTo())


def the_second_arg():
    return CB().with_mods(CBSecondArgIs(), CBRefTo())


def the_type():
    return CB().with_mods(CBTypeIs())

###############################################################################
# MODIFIERS (Subquery Versions)
# - usage |where| operator(
# subquery() ...
# )
###############################################################################


def any_arg_is(subq):
    return any_arg().merge(subq)


def the_receiver_is(subq):
    return the_receiver().merge(subq)


def the_first_arg_is(subq):
    return the_first_arg().merge(subq)


def the_second_arg_is(subq):
    return the_second_arg().merge(subq)


def the_first_args_receiver_is(subq):
    return the_first_args_receiver().merge(subq)


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
    lambda x, y: x.bubbleup().nest(y).insert_and()
)

isa = Infixer(
    lambda x, y: x.merge(y)
)

###############################################################################

label = lambda s: s
label_as_match = lambda: '$match'
