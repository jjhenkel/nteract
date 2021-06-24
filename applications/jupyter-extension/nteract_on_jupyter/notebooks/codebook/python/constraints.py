

class CBName:
    _kind = 'Python.Constraints.Name'

    def __init__(self, value):
        self.kind = CBName._kind
        self.value = value
        self.precedence = 6

    def __str__(self):
        return 'name=`{}`'.format(self.value)

    def to_path(self, path):
        return '"{}"={}'.format(self.value, path)

    @staticmethod
    def kind():
        return CBName._kind


class CBSameText:
    _kind = 'Python.Constraints.SameText'

    def __init__(self, value):
        self.kind = CBSameText._kind
        self.value = value

    def __str__(self):
        return 'text_same_as=`{}`'.format(self.value)

    @staticmethod
    def kind():
        return CBSameText._kind


class CBEveryChildHasType:
    _kind = 'Python.Constraints.EveryChildHasType'

    def __init__(self, value):
        self.kind = CBEveryChildHasType._kind
        self.value = value

    def __str__(self):
        return 'every_child_type=`{}`'.format(self.value)

    @staticmethod
    def kind():
        return CBEveryChildHasType._kind


class CBText:
    _kind = 'Python.Constraints.Text'

    def __init__(self, value):
        self.kind = CBText._kind
        self.value = value
        self.precedence = 7

    def __str__(self):
        return 'text=`{}`'.format(self.value)

    @staticmethod
    def kind():
        return CBText._kind


class CBFieldIndex:
    _kind = 'Python.Constraints.FieldIndex'

    def __init__(self, field, index=None):
        self.kind = CBFieldIndex._kind
        self.field = field
        self.index = index
        assert self.field is not None or self.index is not None
        self.precedence = 8

    def __str__(self):
        if self.field is not None and self.index is not None:
            return '$.f_{}[{}]'.format(self.field, self.index)
        elif self.field is not None:
            return '$.f_{}'.format(self.field)
        elif self.index is not None:
            return '$.[{}]'.format(self.index)
        else:
            assert False

    def to_path(self, path):
        if self.field is not None and self.index is not None:
            return '{}.f_{}[{}]'.format(path, self.field, self.index)
        elif self.field is not None:
            return '{}.f_{}'.format(path, self.field)
        elif self.index is not None:
            return '{}[{}]'.format(path, self.index)
        else:
            assert False

    @staticmethod
    def kind():
        return CBFieldIndex._kind


class CBStepsAway:
    _kind = 'Python.Constraints.StepsAway'

    def __init__(self, steps, op=None):
        self.kind = CBStepsAway._kind
        self.steps = steps
        self.op = op
        self.precedence = 9

    def __str__(self):
        if self.op is None:
            return '$steps_away{}'
        else:
            return '$steps_away{{{}{}}}'.format(self.op, self.steps)

    def to_path(self, path):
        return path + (
            '{}' if self.op is None 
            else '{{{}{}}}'.format(self.op, self.steps)
        )

    @staticmethod
    def kind():
        return CBStepsAway._kind


class CBAllowCastsAndParens:
    _kind = 'Python.Constraints.AllowCastsAndParens'

    def __init__(self, max_depth = None):
        self.kind = CBAllowCastsAndParens._kind
        self.max_depth = max_depth
        self.precedence = -1

    def __str__(self):
        if self.max_depth is None:
            return '$allow_casts_and_parens'
        else:
            return '$allow_casts_and_parens[<={}]'.format(self.max_depth)

    def to_path(self, path):
        return '(' + path + ')'

    @staticmethod
    def kind():
        return CBAllowCastsAndParens._kind



class CBFromSet:
    _fidx = 1
    _kind = 'Python.Constraints.FromSet'

    def __init__(self, value):
        self.kind = CBFromSet._kind
        self.value = value
        self.file = '/app/inputs-{}.csv'.format(CBFromSet._fidx)
        with open(self.file, 'w') as fh:
            fh.write('\n'.join(map(str, self.value)))
        CBFromSet._fidx += 1

    def __str__(self):
        return '$from_set'

    @staticmethod
    def kind():
        return CBFromSet._kind
        