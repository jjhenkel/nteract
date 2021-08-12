
class CBRhs:
    _kind = 'C#.Mods.TheRhs'
    
    def __init__(self):
        self.kind = CBRhs._kind

    def __str__(self):
        return '$the_rhs'

    @staticmethod
    def kind():
        return CBRhs._kind


class CBLhs:
    _kind = 'C#.Mods.TheLhs'
    
    def __init__(self):
        self.kind = CBLhs._kind

    def __str__(self):
        return '$the_lhs'

    @staticmethod
    def kind():
        return CBLhs._kind


class CBMemberIs:
    _kind = 'C#.Mods.MemberIs'
    
    def __init__(self):
        self.kind = CBMemberIs._kind

    def __str__(self):
        return '$member_is'

    @staticmethod
    def kind():
        return CBMemberIs._kind


class CBExpressionIs:
    _kind = 'C#.Mods.ExpressionIs'
    
    def __init__(self):
        self.kind = CBExpressionIs._kind

    def __str__(self):
        return '$expression_is'

    @staticmethod
    def kind():
        return CBExpressionIs._kind


class CBValueIs:
    _kind = 'C#.Mods.ValueIs'
    
    def __init__(self):
        self.kind = CBValueIs._kind

    def __str__(self):
        return '$value_is'

    @staticmethod
    def kind():
        return CBValueIs._kind


class CBBodyIs:
    _kind = 'C#.Mods.BodyIs'
    
    def __init__(self):
        self.kind = CBBodyIs._kind

    def __str__(self):
        return '$body_is'

    @staticmethod
    def kind():
        return CBBodyIs._kind


class CBFunctionIs:
    _kind = 'C#.Mods.FunctionIs'
    
    def __init__(self):
        self.kind = CBFunctionIs._kind

    def __str__(self):
        return '$function_is'

    @staticmethod
    def kind():
        return CBFunctionIs._kind


class CBOnlyLambdaParamIs:
    _kind = 'C#.Mods.OnlyLambdaParamIs'
    
    def __init__(self):
        self.kind = CBOnlyLambdaParamIs._kind

    def __str__(self):
        return '$only_lambda_param_is'

    @staticmethod
    def kind():
        return CBOnlyLambdaParamIs._kind


class CBNameIs:
    _kind = 'C#.Mods.NameIs'
    
    def __init__(self):
        self.kind = CBNameIs._kind

    def __str__(self):
        return '$name_is'

    @staticmethod
    def kind():
        return CBNameIs._kind


class CBFirstArgIs:
    _kind = 'C#.Mods.FirstArgIs'
    
    def __init__(self):
        self.kind = CBFirstArgIs._kind

    def __str__(self):
        return '$first_arg'

    @staticmethod
    def kind():
        return CBFirstArgIs._kind


class CBSecondArgIs:
    _kind = 'C#.Mods.SecondArgIs'
    
    def __init__(self):
        self.kind = CBSecondArgIs._kind

    def __str__(self):
        return '$second_arg'

    @staticmethod
    def kind():
        return CBSecondArgIs._kind


class CBSubscriptIs:
    _kind = 'C#.Mods.SubscriptIs'
    
    def __init__(self):
        self.kind = CBSubscriptIs._kind

    def __str__(self):
        return '$subscript_is'

    @staticmethod
    def kind():
        return CBSubscriptIs._kind


class CBAttributeIs:
    _kind = 'C#.Mods.AttributeIs'
    
    def __init__(self):
        self.kind = CBAttributeIs._kind

    def __str__(self):
        return '$attribute_is'

    @staticmethod
    def kind():
        return CBAttributeIs._kind


class CBObjectIs:
    _kind = 'C#.Mods.ObjectIs'
    
    def __init__(self):
        self.kind = CBObjectIs._kind

    def __str__(self):
        return '$object_is'

    @staticmethod
    def kind():
        return CBObjectIs._kind


class CBOnlySubscriptIs:
    _kind = 'C#.Mods.OnlySubscriptIs'
    
    def __init__(self):
        self.kind = CBOnlySubscriptIs._kind

    def __str__(self):
        return '$only_subscript_is'

    @staticmethod
    def kind():
        return CBOnlySubscriptIs._kind


class CBAnyArgIs:
    _kind = 'C#.Mods.AnyArgIs'
    
    def __init__(self):
        self.kind = CBAnyArgIs._kind

    def __str__(self):
        return '$any_arg_is'

    @staticmethod
    def kind():
        return CBAnyArgIs._kind


class CBAnyParentIs:
    _kind = 'C#.Mods.AnyParentIs'
    
    def __init__(self):
        self.kind = CBAnyParentIs._kind

    def __str__(self):
        return '$any_parent_is'

    @staticmethod
    def kind():
        return CBAnyParentIs._kind


class CBUses:
    _kind = 'C#.Mods.Uses'
    
    def __init__(self):
        self.kind = CBUses._kind

    def __str__(self):
        return '$uses'

    @staticmethod
    def kind():
        return CBUses._kind


class CBFirstArgIs:
    _kind = 'C#.Mods.FirstArgIs'
    
    def __init__(self):
        self.kind = CBFirstArgIs._kind

    def __str__(self):
        return '$first_arg_is'

    @staticmethod
    def kind():
        return CBFirstArgIs._kind


class CBRefTo:
    _kind = 'C#.Mods.RefTo'
    
    def __init__(self):
        self.kind = CBRefTo._kind

    def __str__(self):
        return '$ref_to'

    @staticmethod
    def kind():
        return CBRefTo._kind


class CBEveryChildIs:
    _kind = 'C#.Mods.EveryChildIs'
    
    def __init__(self):
        self.kind = CBEveryChildIs._kind

    def __str__(self):
        return '$every_child'

    @staticmethod
    def kind():
        return CBEveryChildIs._kind


class CBAnyChildIs:
    _kind = 'C#.Mods.AnyChildIs'
    
    def __init__(self):
        self.kind = CBAnyChildIs._kind

    def __str__(self):
        return '$any_child'

    @staticmethod
    def kind():
        return CBAnyChildIs._kind


class CBParentIs:
    _kind = 'C#.Mods.ParentIs'
    
    def __init__(self):
        self.kind = CBParentIs._kind

    def __str__(self):
        return '$parent'

    @staticmethod
    def kind():
        return CBParentIs._kind


class CBChildIs:
    _kind = 'C#.Mods.ChildIs'
    
    def __init__(self):
        self.kind = CBChildIs._kind

    def __str__(self):
        return '$child'

    @staticmethod
    def kind():
        return CBChildIs._kind


class CBFirstChildIs:
    _kind = 'C#.Mods.FirstChildIs'
    
    def __init__(self):
        self.kind = CBFirstChildIs._kind

    def __str__(self):
        return '$first_child'

    @staticmethod
    def kind():
        return CBFirstChildIs._kind


class CBModuleName:
    _kind = 'C#.Mods.ModuleName'
    
    def __init__(self):
        self.kind = CBModuleName._kind

    def __str__(self):
        return '$module_name'

    @staticmethod
    def kind():
        return CBModuleName._kind


class CBImportedName:
    _kind = 'C#.Mods.ImportedName'
    
    def __init__(self):
        self.kind = CBImportedName._kind

    def __str__(self):
        return '$imported_name'

    @staticmethod
    def kind():
        return CBImportedName._kind


class CBModuleRoot:
    _kind = 'C#.Mods.ModuleRoot'
    
    def __init__(self):
        self.kind = CBModuleRoot._kind

    def __str__(self):
        return '$module_root'

    @staticmethod
    def kind():
        return CBModuleRoot._kind


class CBCallTarget:
    _kind = 'C#.Mods.CallTarget'
    
    def __init__(self):
        self.kind = CBCallTarget._kind

    def __str__(self):
        return '$call_target'

    @staticmethod
    def kind():
        return CBCallTarget._kind


class CBFirstChildIs:
    _kind = 'C#.Mods.TheFirstChild'
    
    def __init__(self):
        self.kind = CBFirstChildIs._kind

    def __str__(self):
        return '$the_first_child'

    @staticmethod
    def kind():
        return CBFirstChildIs._kind


class CBSecondChildIs:
    _kind = 'C#.Mods.TheSecondChild'
    
    def __init__(self):
        self.kind = CBSecondChildIs._kind

    def __str__(self):
        return '$the_second_child'

    @staticmethod
    def kind():
        return CBSecondChildIs._kind


class CBThirdChildIs:
    _kind = 'C#.Mods.TheThirdChild'
    
    def __init__(self):
        self.kind = CBThirdChildIs._kind

    def __str__(self):
        return '$the_third_child'

    @staticmethod
    def kind():
        return CBThirdChildIs._kind


class CBNoThirdChild:
    _kind = 'C#.Mods.NoThirdChild'
    
    def __init__(self):
        self.kind = CBNoThirdChild._kind

    def __str__(self):
        return '$no_third_child'

    @staticmethod
    def kind():
        return CBNoThirdChild._kind

