import sys
import time
import dask.dataframe as ddf

from ..python import *
from functools import lru_cache
from .astypemap import ASTYPE_MAP


def _kw_arg(df, n): return self.exec(
    call(from_set(df, 'gid_call')) % 'call'
    | where | any_arg()
    | isa | keyword_argument(with_name(n))
    | where | the_value()
    | isa | literal() % n
).set_index('gid_call')[['source_text_{}'.format(n)]]


def _kw_def(other, n, default): return lambda gid: (
    other.loc[[gid], :]['source_text_{}'.format(n)].values[0] if (
        gid in other.index
    ) else default(gid)
)


class ListStream:
    def __init__(self):
        self.data = []
        self.original_stdout = None

    def write(self, s):
        self.data.append(s)

    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, ext_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        # pass


class DSNotebooks:
    def __init__(self, partition, prefilters=None):
        self.partition = partition
        self.prefilters = prefilters
        self.compile = True
        self.use_dask = False

    def set_prefilters(self, prefilters):
        self.prefilters = prefilters

    def set_partition(self, partition):
        self.partition = partition

    def exec(self, query, extra_prefilters=None):
        prefilters = None
        if self.prefilters is not None:
            prefilters = self.prefilters
        if extra_prefilters is not None:
            if prefilters is not None:
                prefilters = prefilters.intersection(extra_prefilters)
            else:
                prefilters = extra_prefilters

        return execute(
            query,
            compile=self.compile,
            use_dask=self.use_dask,
            partition=self.partition
        )

    @lru_cache(maxsize=None)
    def _get_call_with_args(self):
        """
        This method gets all of the calls of the form `x.foo(...)`
        and associates, with each call, a dictionary of keyword 
        arguments (if they exist) and the first/second argument 
        (if they exist).

        So, roughly, we're doing something like this:
          finds: df.drop('x', inplace=True)
          produces: [ drop, { inplace: True }, 'x', NaN ]
        """
        # First query for calls
        all_calls = self.exec(
            call() % 'call'
            | where | the_function()
            | isa | attribute()
            | where | the_attribute()
            | isa | identifier() % 'name'
        ).set_index('gid_call', drop=False)

        # Then keyword args
        all_keyword_args = self.exec(
            call() % 'call'
            | where | any_arg()
            | isa | keyword_argument()
            | where | the_name_is(identifier() % select_as('text', 'kwa_name'))
            | and_w | the_value_is(literal() % select_as('text', 'kwa_value'))
        ).set_index('gid_call', drop=False)

        # Then "first" arguments
        first_args = self.exec(
            call() % 'call'
            | where | the_first_arg()
            | is_ | anything() % 'arg1'
        ).set_index('gid_call', drop=False)

        # and "second" arguments
        second_args = self.exec(
            call() % 'call'
            | where | the_second_arg()
            | is_ | anything() % 'arg2'
        ).set_index('gid_call', drop=False)

        # Smush this all together into one big frame
        result = all_calls.join(
            all_keyword_args[['kwa_name', 'kwa_value']].groupby(
                all_keyword_args.index).agg(list)
        ).join(
            first_args[['source_text_arg1']]
        ).join(
            second_args[['source_text_arg2']]
        )

        # Make the kw_args more easily usable later
        result['kw_args'] = result.apply(
            lambda x: dict(
                zip(x['kwa_name'], x['kwa_value'])
            ) if type(x['kwa_name']) == list else {},
            axis=1
        )

        return result.set_index('gid_call', drop=False)

    @lru_cache(maxsize=None)
    def _uses_of_sklearn_imports(self):
        """
        This method finds any calls that are direct references 
        of something imported from a module rooted with `sklearn`

        E.g., `from sklearn import x ... x()` or 
        `import sklearn.blah ... blah.foo()` 
        """
        return pd.concat([
            self.exec(
                call() % select('name') % 'use'
                | where | the_function()
                | isa | use_of(
                    imports() % 'import'
                    | where | the_module_root()
                    | is_ | anything(with_text('sklearn'))
                ) % select_as('def_name', 'the_def')
            ),
            self.exec(
                call() % select('name') % 'use'
                | where | the_function()
                | isa | attribute()
                | where | the_object()
                | isa | use_of(
                    imports() % 'import'
                    | where | the_module_root()
                    | is_ | anything(with_text('sklearn'))
                ) % select_as('def_name', 'ignore')
            )
        ])

    def modelled_methods(self):
        """
        This will capture a handful of methods we have "modelled"
        (that is, we have some understanding of the semantics of the 
        method, however basic it may be, and want to add that 
        understanding to the output operators)
        """
        def _arg_drop_def_true(x):
            return (
                '' if 'drop' in x and x['drop'] == 'False'
                else 'Drop'
            )

        def _arg_drop(x):
            return (
                'Drop' if 'drop' in x and x['drop'] == 'True'
                else ''
            )

        def _arg_copy(x):
            return (
                'Copy' if 'copy' in x and x['copy'] == 'True'
                else ''
            )

        def _arg_deep(x):
            return (
                'Deep' if 'deep' in x and x['deep'] == 'True'
                else ''
            )

        def _arg_inplace(x):
            return (
                'Inplace' if 'inplace' in x and x['inplace'] == 'True'
                else ''
            )

        def _arg_axis(x):
            return (
                'Columns' if 'axis' in x and (
                    x['axis'] == '1' or 'columns' in x['axis']
                ) else 'Rows'
            )

        def _arg_sort(x):
            return (
                'Sorted' if 'sort' in x and x['sort'] == 'True'
                else ''
            )

        def _arg_how_join(x):
            return (
                x['how'].strip('\'"`').capitalize() if 'how' in x
                else 'Left'
            )

        def _arg_how_merge(x):
            return (
                x['how'].strip('\'"`').capitalize() if 'how' in x
                else 'Inner'
            )

        def _replace():
            methods = self._get_call_with_args()
            tmp = methods[methods.source_text_name == 'replace'].copy()

            tmp['pretty'] = (
                'Replace' + tmp['kw_args'].apply(_arg_inplace) + '[]'
            )
            tmp['gid'] = tmp.gid_call

            return tmp

        def _as_type():
            methods = self._get_call_with_args()
            tmp = methods[methods.source_text_name == 'astype'].copy()

            tmp['pretty'] = (
                'AsType' + tmp['kw_args'].apply(_arg_copy) + '[' +
                tmp.source_text_arg1.str.strip('\'"`').str.replace('dtype=', '').apply(
                    lambda x: ASTYPE_MAP[x] if x in ASTYPE_MAP else "Unknown"
                )
                + ']'
            )
            tmp['gid'] = tmp.gid_call

            return tmp

        def _apply():
            methods = self._get_call_with_args()
            tmp = methods[methods.source_text_name == 'apply'].copy()

            tmp['pretty'] = (
                'Apply' + tmp['kw_args'].apply(_arg_axis) + '[]'
            )
            tmp['gid'] = tmp.gid_call

            return tmp

        def _drop():
            methods = self._get_call_with_args()
            tmp = methods[methods.source_text_name == 'drop'].copy()

            tmp['pretty'] = (
                'Drop' + tmp['kw_args'].apply(_arg_inplace) +
                tmp['kw_args'].apply(_arg_axis) + '[]'
            )
            tmp['gid'] = tmp.gid_call

            return tmp

        def _dropna():
            methods = self._get_call_with_args()
            tmp = methods[methods.source_text_name == 'dropna'].copy()

            tmp['pretty'] = (
                'DropNa' + tmp['kw_args'].apply(_arg_inplace) +
                tmp['kw_args'].apply(_arg_axis) + '[]'
            )
            tmp['gid'] = tmp.gid_call

            return tmp

        def _fillna():
            methods = self._get_call_with_args()
            tmp = methods[methods.source_text_name == 'fillna'].copy()

            tmp['pretty'] = (
                'FillNa' + tmp['kw_args'].apply(_arg_inplace) +
                tmp['kw_args'].apply(_arg_axis) + '[]'
            )
            tmp['gid'] = tmp.gid_call

            return tmp

        def _reset_index():
            methods = self._get_call_with_args()
            tmp = methods[methods.source_text_name == 'reset_index'].copy()

            tmp['pretty'] = (
                'ResetIndex' + tmp['kw_args'].apply(_arg_inplace) +
                tmp['kw_args'].apply(_arg_drop) + '[]'
            )
            tmp['gid'] = tmp.gid_call

            return tmp

        def _set_index():
            methods = self._get_call_with_args()
            tmp = methods[methods.source_text_name == 'set_index'].copy()

            tmp['pretty'] = (
                'SetIndex' + tmp['kw_args'].apply(_arg_inplace) +
                tmp['kw_args'].apply(_arg_drop_def_true) + '[]'
            )
            tmp['gid'] = tmp.gid_call

            return tmp

        def _reshape():
            methods = self._get_call_with_args()
            tmp = methods[methods.source_text_name == 'reshape'].copy()

            tmp['pretty'] = (
                'Reshape[]'
            )
            tmp['gid'] = tmp.gid_call

            return tmp

        def _as_matrix():
            methods = self._get_call_with_args()
            tmp = methods[methods.source_text_name == 'as_matrix'].copy()

            tmp['pretty'] = (
                'AsMatrix[]'
            )
            tmp['gid'] = tmp.gid_call

            return tmp

        def _get_dummies():
            methods = self._get_call_with_args()
            tmp = methods[methods.source_text_name == 'get_dummies'].copy()

            tmp['pretty'] = (
                'GetDummies[]'
            )
            tmp['gid'] = tmp.gid_call

            return tmp

        def _map():
            methods = self._get_call_with_args()
            tmp = methods[methods.source_text_name == 'map'].copy()

            tmp['pretty'] = (
                'Map[]'
            )
            tmp['gid'] = tmp.gid_call

            return tmp

        def _copy():
            methods = self._get_call_with_args()
            tmp = methods[methods.source_text_name == 'copy'].copy()

            tmp['pretty'] = (
                'Copy' + tmp['kw_args'].apply(_arg_deep) + '[]'
            )
            tmp['gid'] = tmp.gid_call

            return tmp

        def _array():
            methods = self._get_call_with_args()
            tmp = methods[methods.source_text_name == 'array'].copy()

            tmp['pretty'] = (
                'Array' + tmp['kw_args'].apply(_arg_copy) + '[]'
            )
            tmp['gid'] = tmp.gid_call

            return tmp

        def _merge():
            methods = self._get_call_with_args()
            tmp = methods[methods.source_text_name == 'merge'].copy()

            tmp['pretty'] = (
                tmp['kw_args'].apply(_arg_how_merge) + 'Merge' +
                tmp['kw_args'].apply(_arg_sort) + '[]'
            )
            tmp['gid'] = tmp.gid_call

            return tmp

        def _join():
            methods = self._get_call_with_args()
            tmp = methods[methods.source_text_name == 'join'].copy()

            tmp['pretty'] = (
                tmp['kw_args'].apply(_arg_how_join) + 'Join' +
                tmp['kw_args'].apply(_arg_sort) + '[]'
            )
            tmp['gid'] = tmp.gid_call

            return tmp

        return [
            ('CallApply', _apply),
            ('CallAsType', _as_type),
            ('CallReplace', _replace),
            ('CallDrop', _drop),
            ('CallReshape', _reshape),
            ('CallCopy', _copy),
            ('CallArray', _array),
            ('CallAsMatrix', _as_matrix),
            ('CallGetDummies', _get_dummies),
            ('CallMap', _map),
            ('CallDropNa', _dropna),
            ('CallFillNa', _fillna),
            ('CallResetIndex', _reset_index),
            ('CallSetIndex', _set_index),
            ('CallMerge', _merge),
            ('CallJoin', _join),
        ]

    def row_filters(self):
        """
        Here we are looking for uses of `df.loc[...]` where
        the first subscript is a filter expression. These are
        "row filters" and seem fairly popular.

        We are ALSO looking at general ...[ ... <op> <literal> ]
        exptressions as these _may_ be row filters (depending 
        on where they end up)
        """
        def _row_filters():
            filters = pd.concat([
                self.exec(
                    subscript()
                    | where | the_value_is(
                        attribute()
                        | where | the_attribute_is(
                            identifier(with_text('loc'))
                        )
                        | and_w | the_object_is(identifier() % 'df1')
                    )
                    | and_w | the_first_subscript_is(
                        comparison() % 'op'
                        | where | the_first_child_is(
                            x1() | where | x2(
                                identifier(same_text_as('df1')) % 'df2'
                            )
                        )
                    )
                ) for (x1, x2) in [
                    (attribute, the_attribute_is),
                    (subscript, the_value_is)
                ]])

            maybe_filters = self.exec(
                subscript()
                | where | the_only_subscript_is(
                    comparison() % 'op'
                )
            )

            # We'll use this to modify the op name
            filters['extra'] = ''
            maybe_filters['extra'] = 'Maybe'

            filters = pd.concat([
                filters, maybe_filters
            ]).set_index('gid_op', drop=False)

            literal_filters = pd.concat([self.exec(
                comparison(from_set(filters, 'gid_op')) % 'op'
                | where | the_first_child_is(
                    x1() | where | x2(
                        x3() % 'col'
                    )
                )
                | and_w | the_second_child_is(
                    literal() % 'lit'
                )
            ) for (x1, x2, x3) in [
                (attribute, the_attribute_is, identifier),
                (subscript, the_subscript_is, string)
            ]]).set_index('gid_op', drop=False)

            # print(literal_filters)

            literal_filters = literal_filters.join(
                filters[['extra']]
            )

            ops = literal_filters.apply(
                lambda x: x.source_text_op[
                    x.end_col_col-x.start_col_op:x.start_col_lit - x.end_col_op
                ].replace('\\\\n', '').replace('\\n', '').strip().strip(']').strip(),
                axis=1
            )

            OPS_MAP = {
                '==': 'Equal',
                '!=': 'NotEqual',
                '<': 'LessThan',
                '>': 'GreaterThan',
                '<=': 'LessThanEq',
                '>=': 'GreaterThanEq'
            }
            ops = ops.apply(
                lambda x: OPS_MAP[x] if x in OPS_MAP else 'Unknown'
            )

            literal_filters['pretty'] = (
                literal_filters.extra + 'FilterRows' + ops +
                '[' + literal_filters.source_text_col + ',' +
                literal_filters.source_text_lit + ']'
            )
            literal_filters['gid'] = literal_filters.gid_op

            return literal_filters

        return [
            ('FilterRows', _row_filters)
        ]

    def row_projections(self):
        """
        Here we are looking for uses of `df.loc[...]` that deal
        with row projections. In particular, we're looking at 
        cases where users select a single row, several rows, or 
        slice rows.
        """
        def template(inner): return pd.concat([
            self.exec(
                subscript() % 'op'
                | where | the_value_is(
                    attribute()
                    | where | the_attribute_is(
                        identifier(with_text('loc'))
                    )
                )
                | and_w | inner(x)
            ) for x in [string(), integer()]
        ])

        def _row_slices():
            # df.loc['a':'b'...]
            row_slices = template(
                lambda x: the_first_subscript_is(
                    slice_()
                    | where | the_first_child_is(x % 'sb')
                    | and_w | the_second_child_is(x % 'se')
                ),

            )

            row_slices['pretty'] = (
                'ProjectRowSlice[' + row_slices.source_text_sb +
                ':' + row_slices.source_text_se + ']'
            )
            row_slices['gid'] = row_slices.gid_op

            return row_slices

        def _project_rows():
            # df.loc[["a", "b", "c"]...]
            project_rows = template(
                lambda x: the_first_subscript_is(
                    use_of(list_(
                        where_every_child_has_type(x.type)
                    ) % 'rows')
                )
            )

            project_rows['rows'] = Utils.source_list_to_py_list(
                project_rows, 'source_text_rows', use_dask=False
            )
            project_rows['pretty'] = (
                'ProjectRows[' + project_rows.rows + ']'
            )
            project_rows['gid'] = project_rows.gid_op

            return project_rows

        def _project_single_row():
            # df.loc["a"...]
            project_row = template(
                lambda x: the_first_subscript_is(
                    use_of(x % 'row')
                )
            )

            project_row['pretty'] = (
                'ProjectSingleRow[' + project_row.source_text_row + ']'
            )
            project_row['gid'] = project_row.gid_op

            return project_row

        return [
            ('ProjectRowSlice', _row_slices),
            ('ProjectRows', _project_rows),
            ('ProjectSingleRow', _project_single_row)
        ]

    def column_projections(self):
        """
        Here we are looking for uses of `df.loc[..., ]` that deal
        with column projections. In particular, we're looking at 
        cases where users select a single column, several columns, or 
        slice columns.
        """
        def template(inner): return pd.concat([
            self.exec(
                subscript() % 'op'
                | where | the_value_is(
                    attribute()
                    | where | the_attribute_is(
                        identifier(with_text('loc'))
                    )
                )
                | and_w | inner(x)
            ) for x in [string(), integer()]
        ])

        def _col_slices():
            # df.loc[..., 'a':'b']
            col_slices = template(
                lambda x: the_second_subscript_is(
                    slice_()
                    | where | the_first_child_is(x % 'sb')
                    | and_w | the_second_child_is(x % 'se')
                ),

            )

            col_slices['pretty'] = (
                'ProjectColumnSlice[' + col_slices.source_text_sb +
                ':' + col_slices.source_text_se + ']'
            )
            col_slices['gid'] = col_slices.gid_op

            return col_slices

        def _project_cols():
            # df.loc[..., ["a", "b", "c"]]
            project_cols = template(
                lambda x: the_second_subscript_is(
                    use_of(list_(
                        where_every_child_has_type(x.type)
                    ) % 'cols')
                )
            )

            project_cols['rows'] = Utils.source_list_to_py_list(
                project_cols, 'source_text_cols', use_dask=False
            )
            project_cols['pretty'] = (
                'ProjectColumns[' + project_cols.rows + ']'
            )
            project_cols['gid'] = project_cols.gid_op

            return project_cols

        def _project_single_col():
            # df.loc[..., "a"]
            project_col = template(
                lambda x: the_second_subscript_is(
                    use_of(x % 'col')
                )
            )

            project_col['pretty'] = (
                'ProjectSingleColumn[' + project_col.source_text_col + ']'
            )
            project_col['gid'] = project_col.gid_op

            return project_col

        return [
            ('ProjectColumnSlice', _col_slices),
            ('ProjectColumns', _project_cols),
            ('ProjectSingleColumn', _project_single_col)
        ]

    def maybe_projections(self):
        """
        Aside from using .loc, we could also be just using 
        regular indexing... those cases are hard to disambiguate.

        In such cases, if our target is a DataFrame, then we are 
        projecting columns. If our target is a Series, we are 
        then projecting rows. If our target is neither, we're doing
        some other subscripting operation. 
        """
        def _maybe_project_many():
            # ...[["a", "b", "c"]]
            maybe_pm = pd.concat([self.exec(
                subscript() % 'op'
                | where | the_only_subscript_is(
                    use_of(list_(
                        where_every_child_has_type(x.type)
                    ) % 'xs')
                )
            ) for x in [string(), integer()]])

            maybe_pm['rows'] = Utils.source_list_to_py_list(
                maybe_pm, 'source_text_xs', use_dask=False
            )
            maybe_pm['pretty'] = (
                'MaybeProjectMany[' + maybe_pm.rows + ']'
            )
            maybe_pm['gid'] = maybe_pm.gid_op

            return maybe_pm

        def _maybe_project_single():
            # ...['col for frame/row for series?']
            maybe_ps = pd.concat([self.exec(
                subscript() % 'op'
                | where | the_only_subscript_is(
                    use_of(x % 'x')
                )
            ) for x in [string(), integer()]])

            maybe_ps['pretty'] = (
                'MaybeProjectSingle[' + maybe_ps.source_text_x + ']'
            )
            maybe_ps['gid'] = maybe_ps.gid_op

            return maybe_ps

        return [
            ('MaybeProjectMany', _maybe_project_many),
            ('MaybeProjectSingle', _maybe_project_single)
        ]

    def interesting_attributes(self):
        """
        For both DataFrame and Series there are some number of 
        "interesting" attributes we might care about. 

        E.g., DataFrame.columns / Series.shape / Series.values 
        """
        def _interesting_attributes():
            all_attrs = self.exec(
                attribute() % 'op'
                | where | the_attribute()
                | isa | identifier() % 'attr'
            )

            INTERESTING_ATTRIBUTES = {
                'axes': 'GetAttr[Axes]',
                'columns': 'GetAttr[Columns]',
                'dtypes': 'GetAttr[DataTypes]',
                'empty': 'GetAttr[Empty]',
                'index': 'GetAttr[Index]',
                'ndim': 'GetAttr[NumDims]',
                'shape': 'GetAttr[Shape]',
                'size': 'GetAttr[Size]',
                'values': 'GetAttr[Values]',
                'T': 'GetAttr[Transpose]',
                'dtype': 'GetAttr[DataType]',
                'array': 'GetAttr[Array]',
                'nbytes': 'GetAttr[NumBytes]',
                'name': 'GetAttr[Name]',
                'flags': 'GetAttr[Flags]',
                'hasnans': 'GetAttr[HasNaNs]',
                'loc': 'GetAttr[Loc]',
                'iloc': 'GetAttr[ILoc]',
                'at': 'GetAttr[At]',
                'iat': 'GetAttr[IAt]'
            }

            all_attrs['source_text_op'] = all_attrs.source_text_attr.map(
                INTERESTING_ATTRIBUTES
            )

            all_attrs['pretty'] = all_attrs.source_text_op
            all_attrs['gid'] = all_attrs.gid_op

            return all_attrs.dropna()

        return [
            ('InterestingAttr', _interesting_attributes)
        ]

    @lru_cache(maxsize=None)
    def _pandas_methods(self):
        return self.exec(
            call() % select('name') % 'use'
            | where | the_function()
            | isa | attribute()
            | where | the_object()
            | isa | use_of(
                imports() % 'import'
                    | where | the_module_root()
                    | is_ | anything(with_text('pandas'))
            ) % select_as('def_name', 'ignore')
        )

    def pandas_methods(self):
        """
        There are a few pandas.* methods we are interested in.
        Primarily, we are looking for things like `read_csv`
        """
        def _pandas_method(pretty, method):
            def _inner():
                methods = self._pandas_methods()
                my_methods = methods[methods.out_name_use == method].copy()
                my_methods['pretty'] = 'Pandas' + pretty + '[]'
                my_methods['gid'] = my_methods.gid_use
                return my_methods
            return _inner

        return [
            ('Pandas' + pretty, _pandas_method(pretty, method))
            for (pretty, method) in [
                ('ReadCSV', 'read_csv'),
                ('ReadPickle', 'read_pickle'),
                ('ReadFWF', 'read_fwf'),
                ('ReadExcel', 'read_excel'),
                ('ReadHTML', 'read_html'),
                ('ReadXML', 'read_xml'),
                ('ReadHDF', 'read_hdf'),
                ('ReadORC', 'read_orc'),
                ('ReadSAS', 'read_sas'),
                ('ReadSQL', 'read_sql'),
                ('ReadSQLTable', 'read_sql_table'),
                ('ReadSQLQuery', 'read_sql_query'),
                ('ReadGBQ', 'read_gbq'),
                ('ReadStata', 'read_stata'),
                ('ReadParquet', 'read_parquet'),
                ('ReadFeather', 'read_feather'),
                ('ReadJSON', 'read_json'),
                ('ReadTable', 'read_table'),
                ('IsNA', 'isna'),
                ('IsNull', 'isnull'),
                ('NotNA', 'notna'),
                ('NotNull', 'notnull'),
                ('Melt', 'melt'),
                ('Merge', 'merge'),
                ('MergeOrdered', 'merge_ordered'),
                ('MergeAsOf', 'merge_asof'),
                ('Concat', 'concat'),
                ('GetDummies', 'get_dummies'),
                ('Factorize', 'factorize'),
                ('Unique', 'unique'),
                ('Pivot', 'pivot'),
                ('PivotTable', 'pivot_table'),
                ('CreateDataFrame', 'DataFrame'),
                ('CreateSeries', 'Series')
            ]
        ]

    @lru_cache(maxsize=None)
    def _sklearn_methods(self):
        uses = self._uses_of_sklearn_imports()

        return self.exec(
            call() % 'call'
            | where | the_function_is(
                attribute()
                | where | the_attribute()
                | isa | identifier() % 'name'
            )
            | and_w | call_target_is(
                use_of(call(
                    from_set(uses, 'gid_use')
                ) % select('name') % 'use')
            )
        ).set_index('gid_use', drop=False).join(
            uses.set_index('gid_use')[['the_def']]
        )

    @lru_cache(maxsize=None)
    def _train_test_splits(self, include_fits=False):
        methods = self._uses_of_sklearn_imports()

        gids = pd.DataFrame()
        gids[['fid', 'gid']] = methods[
            (methods.out_name_use == "train_test_split") |
            (methods.the_def == "train_test_split")
        ][['fid', 'gid_use']]

        if include_fits:
            gids = gids.append(
                self._sklearn_method('Fit', 'fit')()[['fid', 'gid']]
            )

        return self.exec(
            call(from_set(gids, 'gid')) % 'call'
            | where | the_first_arg() | is_ | anything() % 'value'
            | and_w | the_second_arg() | is_ | anything() % 'label'
        )

    def _sklearn_method(self, pretty, method):
        def _inner():
            methods = self._sklearn_methods()
            # Either we _directly_ used a possibly aliased
            # def and want the OG def name, or we just can
            # take the use name (which will be same as def
            # name in cases where aliasing isn't possible)
            methods['the_def'] = methods['the_def'].fillna(
                methods['out_name_use']
            )
            my_methods = methods[methods.source_text_name == method].copy()
            my_methods['pretty'] = 'Sklearn' + pretty + \
                '[' + my_methods.the_def + ']'
            my_methods['gid'] = my_methods.gid_call
            return my_methods
        return _inner

    def sklearn_methods(self):
        """
        There are several methods from sklearn we are interested in.
        Primarily, we are looking for things like the following:

        ```
        from sklearn... import FooClassifier

        foo = FooClassifier()

        foo.<method>(...)
        ```

        ^^^ From the above, we'd like to find transform/fit/predict/etc.
        """

        def _train_test_split():
            tmp = self._train_test_splits().copy()

            tmp['pretty'] = 'TrainTestSplit[]'
            tmp['gid'] = tmp.gid_call

            return tmp

        def _likely_values():
            tmp = self._train_test_splits(include_fits=True).copy()

            tmp['pretty'] = 'LikelyValues[]'
            tmp['gid'] = tmp.gid_value

            return tmp

        def _likely_labels():
            tmp = self._train_test_splits(include_fits=True).copy()

            tmp['pretty'] = 'LikelyLabels[]'
            tmp['gid'] = tmp.gid_label

            return tmp

        return [
            ('TrainTestSplit', _train_test_split),
            ('LikelyValues', _likely_values),
            ('LikelyLabels', _likely_labels),
        ] + [
            ('Sklearn' + pretty, self._sklearn_method(pretty, method))
            for (pretty, method) in [
                ('Transform', 'transform'),
                ('FitTransform', 'fit_transform'),
                ('Fit', 'fit'),
                ('Predict', 'predict'),
                ('Score', 'score'),
                ('FitPredict', 'fit_predict'),
                ('PredictLogProbabilities', 'predict_log_proba'),
                ('PredictProbabilities', 'predict_proba'),
                ('GetDepth', 'get_depth'),
                ('GetNumLeaves', 'get_n_leaves'),
                ('GetParams', 'get_params'),
                ('SetParams', 'set_params'),
                ('GetPrecision', 'get_precision'),
                ('Mahalanobis', 'mahalanobis'),
                ('ErrorNorm', 'error_norm'),
            ]
        ]

    def column_updates(self):
        """
        We'd like to model cases where people "update" a column.
        This could either be creating an entirely new column/set
        of columns or overwriting/updating existing column(s)

        E.g., df['a'] = df['b'].apply(...)
        """
        def _many_column_updates():
            (_, maybe_many), _ = self.maybe_projections()
            _, (_, many), _ = self.column_projections()

            # Have to include "maybe projections" here to capture
            # everything we might want
            many = pd.concat([
                many(), maybe_many()
            ]).set_index('gid', drop=False)

            updates = self.exec(
                assignment()
                | where | the_lhs()
                | isa | subscript(from_set(many, 'gid')) % 'target'
            ).set_index('gid_target', drop=False).join(
                many[['gid', 'pretty']]
            )

            updates['pretty'] = updates.pretty.str.replace(
                'Project', 'UpdateOrAdd'
            )

            return updates

        def _single_column_updates():
            _, (_, maybe_single) = self.maybe_projections()
            _, _, (_, single) = self.column_projections()

            # Have to include "maybe projections" here to capture
            # everything we might want
            singles = pd.concat([
                single(), maybe_single()
            ]).set_index('gid', drop=False)

            updates = self.exec(
                assignment()
                | where | the_lhs()
                | isa | subscript(from_set(singles, 'gid')) % 'target'
            ).set_index('gid_target', drop=False).join(
                singles[['gid', 'pretty']]
            )

            updates['pretty'] = updates.pretty.str.replace(
                'ProjectSingle', 'UpdateOrAddSingle'
            )

            return updates

        return [
            ('UpdateOrAddColumns', _many_column_updates),
            ('UpdateOrAddSingleColumn', _single_column_updates)
        ]

    def merge_join_inputs(self):
        """
        It'd be nice to know if data flow is coming 
        in as the right or left side of something like 
        an LeftJoin or RightMerge. Here we try to model
        this. 
        """
        def _input_to_merge():
            merges = [
                y for (x, y) in self.modelled_methods()
                if x == 'CallMerge'
            ][0]()

            lefts = self.exec(
                call(from_set(merges, 'gid'))
                | where | the_function()
                | isa | attribute()
                | where | the_object()
                | is_ | anything() % 'left'
            )

            rights = self.exec(
                call(from_set(merges, 'gid'))
                | where | the_first_arg()
                | is_ | anything() % 'right'
            )

            lefts['gid'] = lefts.gid_left
            lefts['pretty'] = 'LeftSideOfMerge[]'

            rights['gid'] = rights.gid_right
            rights['pretty'] = 'RightSideOfMerge[]'

            return pd.concat([
                lefts, rights
            ])

        def _input_to_join():
            joins = [
                y for (x, y) in self.modelled_methods()
                if x == 'CallJoin'
            ][0]()

            lefts = self.exec(
                call(from_set(joins, 'gid'))
                | where | the_function()
                | isa | attribute()
                | where | the_object()
                | is_ | anything() % 'left'
            )

            rights = self.exec(
                call(from_set(joins, 'gid'))
                | where | the_first_arg()
                | is_ | anything() % 'right'
            )

            lefts['gid'] = lefts.gid_left
            lefts['pretty'] = 'LeftSideOfJoin[]'

            rights['gid'] = rights.gid_right
            rights['pretty'] = 'RightSideOfJoin[]'

            return pd.concat([
                lefts, rights
            ])

        return [
            ('MergeInput', _input_to_merge),
            ('JoinInput', _input_to_join)
        ]

    def run_all_operators(self):
        all_ops = sum([
            self.row_filters(),
            self.modelled_methods(),
            self.row_projections(),
            self.column_projections(),
            self.maybe_projections(),
            self.interesting_attributes(),
            self.pandas_methods(),
            self.sklearn_methods(),
            self.column_updates(),
            self.merge_join_inputs()
        ], [])

        all_results = {}
        for name, op in all_ops:
            print('Running `{}`...'.format(name))
            with ListStream() as output:
                # Get the data frame (run query)
                start = time.perf_counter()
                ex_str = ''
                frame = pd.DataFrame()
                try:
                    frame = op()
                    frame['fid'] = frame['fid'].astype('Int64')
                    frame['gid'] = frame['gid'].astype('Int64')
                    frame['pretty'] = frame['pretty'].astype('string')
                    frame['op_group'] = name
                except Exception as ex:
                    ex_str = str(ex)
                    assert False, ex_str
               
                elapsed_time = time.perf_counter() - start

                if name in all_results:
                    assert False, "attempt to overwrite results w/ name {}".format(
                        name)

                # Save some neat results meta
                all_results[name] = {
                    'frame': frame,
                    'name': name,
                    'runtime': f"{elapsed_time:.4f}s",
                    'output': output.data,
                    'ex_str': ex_str
                }

        return all_results

    @lru_cache(maxsize=None)
    def flows_reads_to_fits(self):
        sources = pd.concat([
            y()[['fid', 'gid']] for (x, y) in self.pandas_methods()
            if x.startswith('PandasRead')
        ])
        sinks = pd.concat([
            y()[['fid', 'gid']] for (x, y) in self.sklearn_methods()
            if x.startswith('SklearnFit')
        ])

        return self.exec(
            data_flows(
                source=call(
                    from_set(sources, 'gid')) % select('gid') % 'source',
                sink=call(from_set(sinks, 'gid')) % select('gid') % 'sink'
            ) % 'flow'
        )
