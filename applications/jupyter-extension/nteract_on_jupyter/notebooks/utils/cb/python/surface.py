from .constraints import *
from .modifiers import *
from .infixer import Infixer
from .querybuilder import CB

###############################################################################
# SYNTAX NODES
# - usage type(... constraints ...)
###############################################################################


def attribute(*args):
    return CB('attribute').with_constraints(*args)


def param(*args):
    return CB('formal_parameter').with_constraints(*args)


def call(*args):
    return CB('call').with_constraints(*args)


def string(*args):
    return CB('string').with_constraints(*args)


def identifier(*args):
    return CB('identifier').with_constraints(*args)


def integer(*args):
    return CB('integer').with_constraints(*args)


def function(*args):
    return CB('function_definition').with_constraints(*args)


def comment(*args):
    return CB('comment').with_constraints(*args)


def assignment(*args):
    return CB('assignment').with_constraints(*args)


def if_statement(*args):
    return CB('if_statement').with_constraints(*args)


def try_statement(*args):
    return CB('try_statement').with_constraints(*args)


def while_statement(*args):
    return CB('while_statement').with_constraints(*args)


def lambda_expression(*args):
    return CB('lambda').with_constraints(*args)


def import_statement(*args):
    return CB('import_statement').with_constraints(*args)


def import_from_statement(*args):
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


###############################################################################
# MODIFIERS (Operator Versions)
##  - usage |where| operator()
# |isa| subquery()
###############################################################################


def any_arg():
    return CB().with_mods(CBAnyArgIs(), CBRefTo())


def any_child():
    return CB().with_mods(CBAnyChildIs(), CBRefTo())


def module_name():
    return CB().with_mods(CBFirstChildIs())


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


def any_child_is(subq):
    return any_child().merge(subq)


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
