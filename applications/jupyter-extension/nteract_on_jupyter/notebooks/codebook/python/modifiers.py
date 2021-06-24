
class CBRhs:
    _kind = 'Python.Mods.TheRhs'
    
    def __init__(self):
        self.kind = CBRhs._kind

    def __str__(self):
        return '$the_rhs'

    @staticmethod
    def kind():
        return CBRhs._kind


class CBLhs:
    _kind = 'Python.Mods.TheLhs'
    
    def __init__(self):
        self.kind = CBLhs._kind

    def __str__(self):
        return '$the_lhs'

    @staticmethod
    def kind():
        return CBLhs._kind


class CBValueIs:
    _kind = 'Python.Mods.ValueIs'
    
    def __init__(self):
        self.kind = CBValueIs._kind

    def __str__(self):
        return '$value_is'

    @staticmethod
    def kind():
        return CBValueIs._kind


class CBSubscriptIs:
    _kind = 'Python.Mods.SubscriptIs'
    
    def __init__(self):
        self.kind = CBSubscriptIs._kind

    def __str__(self):
        return '$subscript_is'

    @staticmethod
    def kind():
        return CBSubscriptIs._kind


class CBAttributeIs:
    _kind = 'Python.Mods.AttributeIs'
    
    def __init__(self):
        self.kind = CBAttributeIs._kind

    def __str__(self):
        return '$attribute_is'

    @staticmethod
    def kind():
        return CBAttributeIs._kind


class CBObjectIs:
    _kind = 'Python.Mods.ObjectIs'
    
    def __init__(self):
        self.kind = CBObjectIs._kind

    def __str__(self):
        return '$object_is'

    @staticmethod
    def kind():
        return CBObjectIs._kind


class CBOnlySubscriptIs:
    _kind = 'Python.Mods.OnlySubscriptIs'
    
    def __init__(self):
        self.kind = CBOnlySubscriptIs._kind

    def __str__(self):
        return '$only_subscript_is'

    @staticmethod
    def kind():
        return CBOnlySubscriptIs._kind


class CBAnyArgIs:
    _kind = 'Python.Mods.AnyArgIs'
    
    def __init__(self):
        self.kind = CBAnyArgIs._kind

    def __str__(self):
        return '$any_arg_is'

    @staticmethod
    def kind():
        return CBAnyArgIs._kind


class CBAnyParentIs:
    _kind = 'Python.Mods.AnyParentIs'
    
    def __init__(self):
        self.kind = CBAnyParentIs._kind

    def __str__(self):
        return '$any_parent_is'

    @staticmethod
    def kind():
        return CBAnyParentIs._kind


class CBUses:
    _kind = 'Python.Mods.Uses'
    
    def __init__(self):
        self.kind = CBUses._kind

    def __str__(self):
        return '$uses'

    @staticmethod
    def kind():
        return CBUses._kind


class CBFirstArgIs:
    _kind = 'Python.Mods.FirstArgIs'
    
    def __init__(self):
        self.kind = CBFirstArgIs._kind

    def __str__(self):
        return '$first_arg_is'

    @staticmethod
    def kind():
        return CBFirstArgIs._kind


class CBRefTo:
    _kind = 'Python.Mods.RefTo'
    
    def __init__(self):
        self.kind = CBRefTo._kind

    def __str__(self):
        return '$ref_to'

    @staticmethod
    def kind():
        return CBRefTo._kind


class CBEveryChildIs:
    _kind = 'Python.Mods.EveryChildIs'
    
    def __init__(self):
        self.kind = CBEveryChildIs._kind

    def __str__(self):
        return '$every_child'

    @staticmethod
    def kind():
        return CBEveryChildIs._kind


class CBAnyChildIs:
    _kind = 'Python.Mods.AnyChildIs'
    
    def __init__(self):
        self.kind = CBAnyChildIs._kind

    def __str__(self):
        return '$any_child'

    @staticmethod
    def kind():
        return CBAnyChildIs._kind


class CBChildIs:
    _kind = 'Python.Mods.ChildIs'
    
    def __init__(self):
        self.kind = CBChildIs._kind

    def __str__(self):
        return '$child'

    @staticmethod
    def kind():
        return CBChildIs._kind


class CBFirstChildIs:
    _kind = 'Python.Mods.FirstChildIs'
    
    def __init__(self):
        self.kind = CBFirstChildIs._kind

    def __str__(self):
        return '$first_child'

    @staticmethod
    def kind():
        return CBFirstChildIs._kind


class CBModuleName:
    _kind = 'Python.Mods.ModuleName'
    
    def __init__(self):
        self.kind = CBModuleName._kind

    def __str__(self):
        return '$module_name'

    @staticmethod
    def kind():
        return CBModuleName._kind


class CBImportedName:
    _kind = 'Python.Mods.ImportedName'
    
    def __init__(self):
        self.kind = CBImportedName._kind

    def __str__(self):
        return '$imported_name'

    @staticmethod
    def kind():
        return CBImportedName._kind


class CBModuleRoot:
    _kind = 'Python.Mods.ModuleRoot'
    
    def __init__(self):
        self.kind = CBModuleRoot._kind

    def __str__(self):
        return '$module_root'

    @staticmethod
    def kind():
        return CBModuleRoot._kind


class CBCallTarget:
    _kind = 'Python.Mods.CallTarget'
    
    def __init__(self):
        self.kind = CBCallTarget._kind

    def __str__(self):
        return '$call_target'

    @staticmethod
    def kind():
        return CBCallTarget._kind


class CBFirstChildIs:
    _kind = 'Python.Mods.TheFirstChild'
    
    def __init__(self):
        self.kind = CBFirstChildIs._kind

    def __str__(self):
        return '$the_first_child'

    @staticmethod
    def kind():
        return CBFirstChildIs._kind


class CBSecondChildIs:
    _kind = 'Python.Mods.TheSecondChild'
    
    def __init__(self):
        self.kind = CBSecondChildIs._kind

    def __str__(self):
        return '$the_second_child'

    @staticmethod
    def kind():
        return CBSecondChildIs._kind


class CBThirdChildIs:
    _kind = 'Python.Mods.TheThirdChild'
    
    def __init__(self):
        self.kind = CBThirdChildIs._kind

    def __str__(self):
        return '$the_third_child'

    @staticmethod
    def kind():
        return CBThirdChildIs._kind


class CBNoThirdChild:
    _kind = 'Python.Mods.NoThirdChild'
    
    def __init__(self):
        self.kind = CBNoThirdChild._kind

    def __str__(self):
        return '$no_third_child'

    @staticmethod
    def kind():
        return CBNoThirdChild._kind

