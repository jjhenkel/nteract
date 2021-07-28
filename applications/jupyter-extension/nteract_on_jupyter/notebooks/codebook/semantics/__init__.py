from os import stat
from ..python import *
from functools import lru_cache



def _kw_arg(df, n): return DSNotebooks.exec(
    call(from_set(df, 'gid_call')) % 'call'
    | where | any_arg()
    | isa | keyword_argument(with_name(n))
    | where | the_value()
    | isa | literal() % n
).set_index('gid_call')[['source_text_{}'.format(n)]]

def _kw_def(other, n, default): return lambda gid: (
    other.loc[[gid],:]['source_text_{}'.format(n)].values[0] if (
        gid in other.index
    ) else default(gid)
)


class DSNotebooks:
    _compile = True
    _prefilters = None

    @staticmethod
    def set_prefilters(prefilters):
        DSNotebooks._prefilters = prefilters

    @staticmethod
    def exec(query, extra_prefilters=None):
        prefilters = None
        if DSNotebooks._prefilters is not None:
            prefilters = DSNotebooks._prefilters
        if extra_prefilters is not None:
            if prefilters is not None:
                prefilters = prefilters.intersection(extra_prefilters)
            else:
                prefilters = extra_prefilters
    
        return execute(
            query,
            compile=DSNotebooks._compile,
            prefilter_files=prefilters
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def sklearn_imports():
        df1 = DSNotebooks.exec(
            imports() % 'import'
            | where | the_module_root()
            | is_ | anything(with_text('sklearn'))
        )

        # df2 = DSNotebooks.exec(
        #     imports(
        #         from_set(df1, 'gid_import')
        #     ) % select('name') % 'import'
        # ).set_index('gid_import')

        # df1['out_name_import'] = df1.gid_import.apply(
        #     lambda x: df2.loc[x].out_name_import
        # )

        return df1

    @staticmethod
    @lru_cache(maxsize=None)
    def uses_of_sklearn_imports():
        df = DSNotebooks.exec(
            call() % 'use'
            | where | call_target()
            | isa |  use_of(imports(
                from_set(DSNotebooks.sklearn_imports(), 'gid_import')
            ) % 'import')
        )

        return df

    @staticmethod
    @lru_cache(maxsize=None)
    def sklearn_transforms():
        temp = DSNotebooks.exec(
            call(with_name('transform')) % 'call'
        )

        df = DSNotebooks.exec(
            call(from_set(temp, 'gid_call')) % 'transform'
            | where | call_target()
            | is_ | use_of(call(
                from_set(DSNotebooks.uses_of_sklearn_imports(), 'gid_use')
            ) % select('name') % 'use')
        )

        df['pretty'] = 'Transform[' + df.out_name_use + ']'
        df['gid'] = df.gid_transform

        return df

    @staticmethod
    @lru_cache(maxsize=None)
    def sklearn_fit_transforms():
        temp = DSNotebooks.exec(
            call(with_name('fit_transform')) % 'call'
        )

        df = DSNotebooks.exec(
            call(from_set(temp, 'gid_call')) % 'transform'
            | where | call_target()
            | is_ | use_of(call(
                from_set(DSNotebooks.uses_of_sklearn_imports(), 'gid_use')
            ) % select('name') % 'use')
        )

        df['pretty'] = 'FitTransform[' + df.out_name_use + ']'
        df['gid'] = df.gid_transform

        return df

    @staticmethod
    @lru_cache(maxsize=None)
    def sklearn_fits():
        temp = DSNotebooks.exec(
            call(with_name('fit')) % 'call'
        )

        df = DSNotebooks.exec(
            call(from_set(temp, 'gid_call')) % 'fit'
            | where | call_target()
            | is_ | use_of(call(
                from_set(DSNotebooks.uses_of_sklearn_imports(), 'gid_use')
            ) % select('name') % 'use')
        )

        df['pretty'] = 'Fit[' + df.out_name_use + ']'
        df['gid'] = df.gid_fit

        return df

    @staticmethod
    @lru_cache(maxsize=None)
    def column_additions():
        temp = DSNotebooks.single_col_projections()

        df = DSNotebooks.exec(
            assignment()
            | where | the_lhs()
            | isa | subscript(from_set(temp, 'gid')) % 'target'
        )

        df['pretty'] = df.gid_target.apply(
            lambda x: 'AddCol[' + str(temp[temp.gid ==
                                       x].cols.str.join(',').values[0]) + ']'
        )
        df['gid'] = df.gid_target

        return df

    @staticmethod
    @lru_cache(maxsize=None)
    def likely_values():
        temp = DSNotebooks.sklearn_fits()

        df1 = DSNotebooks.exec(
            call(from_set(temp, 'gid_fit')) % 'fit'
            | where | the_first_arg()
            | is_ | anything() % 'arg'
        )

        df1['pretty'] = 'Likely[Values]'
        df1['gid'] = df1.gid_arg

        temp = DSNotebooks.train_test_splits()

        df2 = pd.DataFrame()
        df2['gid'] = temp.gid_value
        df2['pretty'] = 'Likely[Values]'

        return pd.concat([
            df1[['gid', 'pretty']],
            df2[['gid', 'pretty']],
        ])

    @staticmethod
    @lru_cache(maxsize=None)
    def likely_labels():
        temp = DSNotebooks.sklearn_fits()

        df1 = DSNotebooks.exec(
            call(from_set(temp, 'gid_fit')) % 'fit'
            | where | the_second_arg()
            | is_ | anything() % 'arg'
        )

        df1['pretty'] = 'Likely[Labels]'
        df1['gid'] = df1.gid_arg

        temp = DSNotebooks.train_test_splits()

        df2 = pd.DataFrame()
        df2['gid'] = temp.gid_label
        df2['pretty'] = 'Likely[Labels]'

        return pd.concat([
            df1[['gid', 'pretty']],
            df2[['gid', 'pretty']],
        ])

    @staticmethod
    @lru_cache(maxsize=None)
    def pandas_reads():
        read_calls = [
            'read_csv',
            'read_excel',
            'read_fwf',
            'read_json',
            'read_pickle',
            'read_sql',
            'read_table'
        ]

        temp1 = execute(
            imports(with_name("pandas")) % 'import',
            compile=True
        )

        temp2 = execute(
            use_of(imports(from_set(temp1, 'gid_import'))) % 'use',
            compile=True
        )

        df = execute(
            call() % select('name') % 'call'
            |where| call_target() |is_| anything(from_set(temp2, 'gid_use')),
            compile=True
        )

        df = df[df.out_name_call.isin(read_calls)]

        df['pretty'] = 'Read[' + df.out_name_call + ']'
        df['gid'] = df.gid_call

        return df

    @staticmethod
    @lru_cache(maxsize=None)
    def use_of_pandas_read():
        df = DSNotebooks.exec(
            use_of(call(
                from_set(DSNotebooks.pandas_reads(), 'gid')
            )) % 'use'
        )

        df['gid'] = df.gid_use

        return df

    @staticmethod
    @lru_cache(maxsize=None)
    def single_col_projections():
        tmp1 = DSNotebooks.exec(
            use_of(string() % 'cols') % 'use'
        ).set_index('gid_use', drop=False)
        
        df = DSNotebooks.exec(
            subscript() % 'target'
            | where | the_only_subscript_is(
                anything(from_set(tmp1, 'gid_use')) % 'use'
            )
        )

        df = df.set_index('gid_use').join(tmp1[['source_text_cols']])

        df['cols'] = Utils.source_list_to_py_list(
            df, 'source_text_cols'
        )
        df['pretty'] = 'Project[' + df.cols.str.join(',') + ']'
        df['gid'] = df.gid_target

        return df

    @staticmethod
    @lru_cache(maxsize=None)
    def multi_col_projections():
        tmp0 = DSNotebooks.exec(
            list_(where_every_child_has_type('string')) % 'cols'
        )

        tmp1 = DSNotebooks.exec(
            use_of(list_(from_set(tmp0, 'gid_cols')) % 'cols') % 'use'
        ).set_index('gid_use', drop=False)


        df = DSNotebooks.exec(
            subscript() % 'target'
            | where | the_only_subscript_is(
                anything(from_set(tmp1, 'gid_use')) % 'use'
            )
        )

        df = df.set_index('gid_use').join(
            tmp1[['source_text_cols']]
        )

        df['cols'] = Utils.source_list_to_py_list(
            df, 'source_text_cols'
        )
        df['pretty'] = 'Project[' + df.cols.str.join(',') + ']'
        df['gid'] = df.gid_target

        return df

    @staticmethod
    @lru_cache(maxsize=None)
    def projections():
        additions = DSNotebooks.column_additions()

        temp = pd.concat([
            DSNotebooks.single_col_projections(),
            DSNotebooks.multi_col_projections()
        ])

        # Return the ones that aren't part of AddCol
        return temp[~temp.gid.isin(additions.gid_target)]

    @staticmethod
    @lru_cache(maxsize=None)
    def flows_reads_to_fits():
        df = DSNotebooks.exec(
            data_flows(
                source=call(
                    from_set(DSNotebooks.pandas_reads(), 'gid')) % select('gid') % 'source',
                sink=call(from_set(DSNotebooks.sklearn_fits(), 'gid')) % select('gid') % 'sink'
            ) % 'flow'
        )

        return df

    @staticmethod
    @lru_cache(maxsize=None)
    def filters():
        filters = DSNotebooks.exec(
            subscript(with_exactly_two_children()) % 'expr'
            | where | the_subscript_is(comparison())
        )

        filters['gid'] = filters.gid_expr
        filters['pretty'] = 'Filter[]'

        return filters

    @staticmethod
    @lru_cache(maxsize=None)
    def single_compares():
        compares = DSNotebooks.exec(
            comparison(with_exactly_two_children()) % 'filter'
            | where | the_first_child_is( anything() % 'lhs' )
            | and_w | the_second_child_is( literal() % 'rhs' )
        )

        compares['op'] = Utils.get_comp_op(compares, 'source_text_filter', 'source_text_lhs', 'source_text_rhs')
        compares['rhs_type'] = Utils.get_literal_type(compares, 'source_text_rhs')

        uses_of_compares = DSNotebooks.exec(
            use_of(comparison(from_set(compares, 'gid_filter')) % 'comp') % 'use'
        )

        uses_of_compares = uses_of_compares.set_index('gid_comp').join(
            compares.set_index('gid_filter')[['op', 'rhs_type']]
        )
        uses_of_compares['gid'] = uses_of_compares.gid_use
        uses_of_compares['pretty'] = 'Compare[' + uses_of_compares.op + ',' + uses_of_compares.rhs_type + ']'

        return uses_of_compares



    @staticmethod
    @lru_cache(maxsize=None)
    def drops():
        drops = DSNotebooks.exec(
            call(with_name('drop')) % 'call'
        )
        # files_with_drops = set(drops.fpath.unique())

        # lists1 = DSNotebooks.exec(
        #     list_(where_every_child_has_type('string')) % 'cols',
        #     extra_prefilters=files_with_drops
        # )
        # lists2 = DSNotebooks.exec(
        #     list_(where_every_child_has_type('integer')) % 'col_ids',
        #     extra_prefilters=files_with_drops
        # )
        # uses1 = DSNotebooks.exec(
        #     use_of(
        #         string() % 'cols'
        #     ) % 'use',
        #     extra_prefilters=files_with_drops
        # )
        # uses2 = DSNotebooks.exec(
        #     use_of(
        #         list_(from_set(lists1, 'gid_cols')) % 'cols'
        #     ) % 'use',
        #     extra_prefilters=files_with_drops
        # )
        # uses3 = DSNotebooks.exec(
        #     use_of(
        #         list_(from_set(lists2, 'gid_col_ids')) % 'col_ids'
        #     ) % 'use',
        #     extra_prefilters=files_with_drops
        # )

        # df1 = DSNotebooks.exec(
        #     call(from_set(drops, 'gid_call')) % 'call'
        #     | where | the_first_arg()
        #     | isa | anything(from_set(uses1, 'gid_use')) % 'use',
        #     extra_prefilters=files_with_drops
        # )
        # df2 = DSNotebooks.exec(
        #     call(from_set(drops, 'gid_call')) % 'call'
        #     | where | the_first_arg()
        #     | is_ | anything(from_set(uses2, 'gid_use')) % 'use',
        #     extra_prefilters=files_with_drops
        # )
        # df3 = DSNotebooks.exec(
        #     call(from_set(drops, 'gid_call')) % 'call'
        #     | where | the_first_arg()
        #     | isa | subscript()
        #     | where | the_value_is(
        #         attribute()
        #         | where | the_attribute()
        #         | isa | identifier(with_text('columns'))
        #     )
        #     | and_w | the_subscript_is(
        #         anything(from_set(uses3, 'gid_use')) % 'use'
        #     ),
        #     extra_prefilters=files_with_drops
        # )

        drops = drops.set_index('gid_call', drop=False)
        drops['gid'] = drops.gid_call

        # uses1 = uses1.set_index('gid_use')
        # uses2 = uses2.set_index('gid_use')
        # uses3 = uses3.set_index('gid_use')

        # df1 = df1.set_index('gid_use').join(uses1[["source_text_cols"]])
        # df1['cols'] = '["' + df1.source_text_cols.str.strip('\'"`') + '"]'
        # df1['cols'] = Utils.source_list_to_py_list(df1, 'source_text_cols')

        # df2 = df2.set_index('gid_use').join(uses2[["source_text_cols"]])
        # df2['cols'] = Utils.source_list_to_py_list(df2, 'source_text_cols')

        # df3 = df3.set_index('gid_use').join(uses3[["source_text_col_ids"]])
        # df3['cols'] = Utils.source_list_to_py_list(df3, 'source_text_col_ids')

        drops = drops.join(_kw_arg(drops, 'axis')).fillna(0)
        drops = drops.join(_kw_arg(drops, 'level')).fillna("None")
        drops = drops.join(_kw_arg(drops, 'index')).fillna("None")
        drops = drops.join(_kw_arg(drops, 'labels')).fillna("None")
        drops = drops.join(_kw_arg(drops, 'inplace')).fillna(False)
        drops = drops.join(_kw_arg(drops, 'errors')).fillna("raise")

        drops['pretty'] = (
            'Drop' + (drops.source_text_axis.apply(
                lambda x: "Rows" if x == "0.0" or x == "index" else (
                    "Cols" if x == "1.0" or x == "columns" else ""
                )
            )) + (drops.source_text_inplace.apply(
                lambda x: "Inplace" if x else ""
            )) + '[level=' + drops.source_text_level + ';errors=' + drops.source_text_errors.str.strip('\'"`') + ']' 
        )

        return drops

    @staticmethod
    @lru_cache(maxsize=None)
    def train_test_splits():
        df1 = DSNotebooks.exec(
            call(with_name('train_test_split')) % 'call'
            | where | call_target()
            | isa | use_of(imports(from_set(
                DSNotebooks.sklearn_imports(), 'gid_import'
            )))
            | and_w | the_first_arg() | is_ | anything() % 'value'
            | and_w | the_second_arg() | is_ | anything() % 'label'
        )

        df1['test_size'] = df1.gid_call.apply(
            _kw_def(_kw_arg(df1, 'test_size'), 'test_size', lambda x: '0.25')
        ).apply(float)

        df1['train_size'] = df1.gid_call.apply(
            _kw_def(_kw_arg(df1, 'train_size'), 'train_size', lambda x: str(
                1.0 - df1[df1.gid_call == x].test_size.values[0]
            ))
        ).apply(float)

        df1['random_state'] = df1.gid_call.apply(
            _kw_def(_kw_arg(df1, 'random_state'), 'random_state', lambda x: 'None')
        )

        df1['shuffle'] = df1.gid_call.apply(
            _kw_def(_kw_arg(df1, 'shuffle'), 'shuffle', lambda x: 'True')
        ).apply(bool)

        df1['gid'] = df1.gid_call
        df1['pretty'] = (
            'TTSplit[' + df1.train_size.map('{:.2f}'.format) + ',' + df1.test_size.map(
                '{:.2f}'.format) + ',' + df1.random_state.map(str) + ',' + df1.shuffle.map(str) + ']'
        )

        return df1

    @staticmethod
    @lru_cache(maxsize=None)
    def copies():
        copies = DSNotebooks.exec(
            call(with_name('copy')) % 'call'
        )

        copies.set_index('gid_call', drop=False, inplace=True)
        copies['gid'] = copies.gid_call
       
        copies = copies.join(_kw_arg(copies, 'deep')).fillna(True)

        copies['pretty'] = (
            'Copy[deep=' + copies.source_text_deep.apply(str) + ']'
        )

        return copies

    @staticmethod
    @lru_cache(maxsize=None)
    def reshape():
        reshape = DSNotebooks.exec(
            call(with_name('reshape')) % 'call'
        )

        reshape.set_index('gid_call', drop=False, inplace=True)
        reshape['gid'] = reshape.gid_call
       
        reshape['pretty'] = ('Reshape[]')

        return reshape

    @staticmethod
    @lru_cache(maxsize=None)
    def array():
        array = DSNotebooks.exec(
            call(with_name('array')) % 'call'
        )

        array.set_index('gid_call', drop=False, inplace=True)
        array['gid'] = array.gid_call
       
        array = array.join(_kw_arg(array, 'copy')).fillna(True)

        array['pretty'] = (
            'Array[copy=' + array.source_text_copy.apply(str) + ']'
        )

        return array

    @staticmethod
    @lru_cache(maxsize=None)
    def dropna():
        dropna = DSNotebooks.exec(
            call(with_name('dropna')) % 'call'
        )

        dropna.set_index('gid_call', drop=False, inplace=True)
        dropna['gid'] = dropna.gid_call

        dropna = dropna.join(_kw_arg(dropna, 'axis')).fillna(0)
        dropna = dropna.join(_kw_arg(dropna, 'how')).fillna("any")
        dropna = dropna.join(_kw_arg(dropna, 'thresh')).fillna("None")
        dropna = dropna.join(_kw_arg(dropna, 'inplace')).fillna(False)

        dropna['pretty'] = (
            'DropNa' + (dropna.source_text_axis.apply(
                lambda x: "Rows" if x == "0.0" or x == "index" else (
                    "Cols" if x == "1.0" or x == "columns" else ""
                )
            )) + (dropna.source_text_inplace.apply(
                lambda x: "Inplace" if x else ""
            )) + 
            '[how=' + 
            dropna.source_text_how.apply(str).str.strip('\'"`') + 
            ';thresh=' +
            dropna.source_text_thresh.apply(str) +
            ']'
        )

        return dropna

    @staticmethod
    @lru_cache(maxsize=None)
    def applies():
        applies = DSNotebooks.exec(
            call(with_name('apply')) % 'call'
        )

        applies.set_index('gid_call', drop=False, inplace=True)
        applies['gid'] = applies.gid_call

        applies = applies.join(_kw_arg(applies, 'axis')).fillna(0)
        applies = applies.join(_kw_arg(applies, 'raw')).fillna(False)
        applies = applies.join(_kw_arg(applies, 'result_type')).fillna("None")

        applies['pretty'] = (
            'Apply' + (applies.source_text_axis.apply(
                lambda x: "Rows" if x == "0.0" or x == "index" else (
                    "Cols" if x == "1.0" or x == "columns" else ""
                )
            )) + 
            '[raw=' + 
            applies.source_text_raw.apply(str) + 
            ';result_type=' +
            applies.source_text_result_type.apply(str).str.strip('\'"`') +
            ']'
        )

        return applies

    @staticmethod
    @lru_cache(maxsize=None)
    def as_matrix():
        as_matrix = DSNotebooks.exec(
            call(with_name('as_matrix')) % 'call'
        )

        as_matrix.set_index('gid_call', drop=False, inplace=True)
        as_matrix['gid'] = as_matrix.gid_call
       
        # as_matrix = as_matrix.join(_kw_arg(as_matrix, 'deep')).fillna(True)

        as_matrix['pretty'] = (
            'AsMatrix[]'
        )

        return as_matrix

    @staticmethod
    @lru_cache(maxsize=None)
    def maps():
        maps = DSNotebooks.exec(
            call(with_name('map')) % 'call'
        )

        maps.set_index('gid_call', drop=False, inplace=True)
        maps['gid'] = maps.gid_call
       
        # maps = maps.join(_kw_arg(maps, 'deep')).fillna(True)

        maps['pretty'] = (
            'Map[]'
        )

        return maps

    @staticmethod
    @lru_cache(maxsize=None)
    def get_dummies():
        get_dummies = DSNotebooks.exec(
            call(with_name('get_dummies')) % 'call'
        )

        get_dummies.set_index('gid_call', drop=False, inplace=True)
        get_dummies['gid'] = get_dummies.gid_call
       
        # get_dummies = get_dummies.join(_kw_arg(get_dummies, 'deep')).fillna(True)

        get_dummies['pretty'] = (
            'GetDummies[]'
        )

        return get_dummies

    @staticmethod
    @lru_cache(maxsize=None)
    def new_data_frame():
        new_data_frame = DSNotebooks.exec(
            call(with_name('DataFrame')) % 'call'
        )

        new_data_frame.set_index('gid_call', drop=False, inplace=True)
        new_data_frame['gid'] = new_data_frame.gid_call
       
        # new_data_frame = new_data_frame.join(_kw_arg(new_data_frame, 'deep')).fillna(True)

        new_data_frame['pretty'] = (
            'NewDataFrame[]'
        )

        return new_data_frame

    @staticmethod
    @lru_cache(maxsize=None)
    def fillna():
        fillna = DSNotebooks.exec(
            call(with_name('fillna')) % 'call'
        )

        fillna.set_index('gid_call', drop=False, inplace=True)
        fillna['gid'] = fillna.gid_call
       
        fillna = fillna.join(_kw_arg(fillna, 'axis')).fillna("None")
        fillna = fillna.join(_kw_arg(fillna, 'inplace')).fillna(False)
        fillna = fillna.join(_kw_arg(fillna, 'method')).fillna("None")
        fillna = fillna.join(_kw_arg(fillna, 'limit')).fillna("None")

        fillna['pretty'] = (
            'FillNa'+ (fillna.source_text_axis.apply(
                lambda x: "Rows" if x == "0.0" or x == "index" else (
                    "Cols" if x == "1.0" or x == "columns" else ""
                )
            )) + (fillna.source_text_inplace.apply(
                lambda x: "Inplace" if x else ""
            )) + '[limit=' + fillna.source_text_limit.apply(str) + ';method=' + fillna.source_text_method.str.strip('\'"`') + ']'
        )

        return fillna

    @staticmethod
    @lru_cache(maxsize=None)
    def replace():
        replace = DSNotebooks.exec(
            call(with_name('replace')) % 'call'
        )

        replace.set_index('gid_call', drop=False, inplace=True)
        replace['gid'] = replace.gid_call
       
        replace = replace.join(_kw_arg(replace, 'inplace')).fillna(False)
        replace = replace.join(_kw_arg(replace, 'method')).fillna("None")
        replace = replace.join(_kw_arg(replace, 'limit')).fillna("None")

        replace['pretty'] = (
            'Replace' + (replace.source_text_inplace.apply(
                lambda x: "Inplace" if x else ""
            )) + '[limit=' + replace.source_text_limit.apply(str) + ';method=' + replace.source_text_method.str.strip('\'"`') + ']'
        )

        return replace

    @staticmethod
    @lru_cache(maxsize=None)
    def astype():
        TYPE_MAP = {
            'int': 'Int',
            'float': 'Float',
            'str': 'String',
            'np.float32': 'Float32',
            'float32': 'Float32',
            'uint8': 'UInt8',
            'np.float': 'Float',
            'category': 'Categorical',
            'np.uint8': 'UInt8',
            'np.int': 'Int',
            'np.float64': 'Float64',
            'bool': 'Bool',
            'float64': 'Float64',
            'np.int32': 'Int32',
            'np.int64': 'Int64',
            'int32': 'Int32',
            'int64': 'Int64',
            'np.int16': 'Int16',
            'object': 'Object',
            'theano.config.floatX': 'Float',
            'np.int8': 'Int8',
            'np.bool': 'Bool',
            'numpy.float32': 'Float32',
            'np.int_': 'Int',
            'timedelta64[D]': 'TimeDelta',
            'timedelta64[s]': 'TimeDelta',
            'timedelta64[m]': 'TimeDelta',
            'timedelta64[h]': 'TimeDelta',
            'datetime64[ns]': 'DateTime',
            'datetime64[D]': 'DateTime',
            'np.uint64': 'UInt64',
            'f': 'Float',
            'np.double': 'Double',
            'int64': 'Int16',
            'U': 'Unicode',
            '?': 'Bool',
            'b': 'Int8',
            'B': 'UInt8',
            'i': 'Int',
            'u': 'UInt',
            'c': 'ComplexFloat',
            'm': 'TimeDelta',
            'M': 'DateTime',
            'O': 'Object',
            'V': 'RawData',
            'int16': 'Int16',
            'datetime64': 'DateTime',
            'np.float16': 'Float16',
            'int8': 'Int8',
            'double': 'Double',
            'np.uint16': 'UInt16',
            'string': 'String',
            'np.str': 'String',
            'complex': 'Complex',
            'np.complex64': 'Complex',
            'datetime': 'DateTime',
            'dtm.datetime': 'DateTime',
            'datetime.datetime': 'DateTime',
            'numpy.uint8': 'UInt8',
            'uint32': 'UInt32',
            'numpy.float64': 'Float64',
            'uint16': 'UInt16',
            'float16': 'Float16',
            'np.uint32': 'UInt32',
            'unicode': 'Unicode',
            'numpy.int32': 'Int32',
            'numpy.float': 'Float',
            'numpy.int': 'Int'
        }

        astype = DSNotebooks.exec(
            call(with_name('astype')) % 'call'
            |where| the_first_arg()
              |is_| anything() % 'arg'
        )

        astype.set_index('gid_call', drop=False, inplace=True)
        astype['gid'] = astype.gid_call
       
        astype = astype.join(_kw_arg(astype, 'copy')).fillna(True)
        astype = astype.join(_kw_arg(astype, 'errors')).fillna("raise")

        astype['pretty'] = (
            'AsType' + (astype.source_text_arg.str.strip('\'"`').str.replace('dtype=', '').apply(
                lambda x: TYPE_MAP[x] if x in TYPE_MAP else "Unknown"
            )) + '[copy=' + astype.source_text_copy.apply(str) + ';errors=' + astype.source_text_errors.str.strip('\'"`') + ']'
        )

        return astype

    @staticmethod
    @lru_cache(maxsize=None)
    def reset_indices():
        reset_indices = DSNotebooks.exec(
            call(with_name('reset_index')) % 'call'
        )

        reset_indices.set_index('gid_call', drop=False, inplace=True)
        reset_indices['gid'] = reset_indices.gid_call
       
        reset_indices = reset_indices.join(_kw_arg(reset_indices, 'inplace')).fillna(False)
        reset_indices = reset_indices.join(_kw_arg(reset_indices, 'drop')).fillna(False)
        reset_indices = reset_indices.join(_kw_arg(reset_indices, 'level')).fillna("None")

        reset_indices['pretty'] = (
            'ResetIndices' + (reset_indices.source_text_inplace.apply(
                lambda x: "Inplace" if x else ""
            )) + '[drop=' + reset_indices.source_text_drop.apply(str) + ';level=' + reset_indices.source_text_level.str.strip('\'"`') + ']'
        )

        return reset_indices

    @staticmethod
    @lru_cache(maxsize=None)
    def merge():
        merge = DSNotebooks.exec(
            call(with_name('merge')) % 'call'
        )

        merge.set_index('gid_call', drop=False, inplace=True)
        merge['gid'] = merge.gid_call
       
        merge = merge.join(_kw_arg(merge, 'sort')).fillna(False)
        merge = merge.join(_kw_arg(merge, 'how')).fillna("inner")

        merge['pretty'] = (
            'Merge' + (merge.source_text_how.str.strip('\'"`').apply(
                lambda x: x.capitalize()
            )) + '[sort=' + merge.source_text_sort.apply(str) + ']'
        )

        return merge

    @staticmethod
    @lru_cache(maxsize=None)
    def join():
        join = DSNotebooks.exec(
            call(with_name('join')) % 'call'
        )

        join.set_index('gid_call', drop=False, inplace=True)
        join['gid'] = join.gid_call
       
        join = join.join(_kw_arg(join, 'sort')).fillna(False)
        join = join.join(_kw_arg(join, 'how')).fillna("left")

        join['pretty'] = (
            'Join' + (join.source_text_how.str.strip('\'"`').apply(
                lambda x: x.capitalize()
            )) + '[sort=' + join.source_text_sort.apply(str) + ']'
        )

        return join

    @staticmethod
    @lru_cache(maxsize=None)
    def maybe_project():
        maybe_project = DSNotebooks.exec(
            attribute() % 'attr'
            |where| the_attribute()
              |isa| identifier() % 'field'
        )

        maybe_project['gid'] = maybe_project.gid_attr
        maybe_project['pretty'] = 'MaybeProject[' + maybe_project.source_text_field + ']'

        return maybe_project

