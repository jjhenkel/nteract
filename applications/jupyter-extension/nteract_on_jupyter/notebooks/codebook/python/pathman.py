from genericpath import isfile
import os
import glob

# We need to manage a few things here related to file paths
# (1) we need to manage query inputs (gid/fid)
# (2) we need to manage query outputs (parquets)
# (3) we need to manage query targets (input dataset/chunks)

# As we do all this, we should be careful to "namespace" things 
# on both the _session_ ID and the _query_ ID (hash) so that 
# we can have two people, running two notebooks, and two separate
# (perhaps equivalent) queries at the same time without conflicts


class Pathman:
    def __init__(self, dataset, session_id):
        self.dataset = dataset
        self.session_id = session_id
        self.query_hash = None

        self.tmp_dir = '/tmp'
        self.app_dir = '/app'
        self.q_src = os.path.join(self.tmp_dir, 'queries')
        self.q_out = os.path.join(self.tmp_dir, 'query-outputs')
        self.q_in = os.path.join(self.tmp_dir, 'query-inputs')
        self.q_stats = os.path.join(self.tmp_dir, 'query-stats')

        self.souffle_sh = os.path.join(self.app_dir, 'souffle.sh')
        self.wrapper_sh = os.path.join(self.app_dir, 'wrapper.sh')
        self.souffle_comp = '/usr/local/bin/souffle-compile'

        self.client_dir = 'applications/jupyter-extension/nteract_on_jupyter'
        self.notebooks_dir = os.path.join(self.app_dir, self.client_dir, 'notebooks')
        self.api_dir = os.path.join(self.notebooks_dir, 'codebook/python')
        self.snippets_dir = os.path.join(self.api_dir, 'snippets')

        self.target_chunks = list(sorted(glob.glob(
            os.path.join(self.dataset, 'chunked/chunk-*/relations')
        )))


    def get_target_chunks(self, limit=None):
        if limit is None:
            return self.target_chunks
        
        assert limit < len(self.target_chunks), \
            "Limit must be less than {}".format(len(self.target_chunks))
        return self.target_chunks[:limit]

    def set_query_hash(self, query_hash):
        self.query_hash = query_hash

        os.makedirs(os.path.join(
            self.q_in, self.session_id, self.query_hash
        ), exist_ok=True)

        os.makedirs(os.path.join(
            self.q_out, self.session_id, self.query_hash
        ), exist_ok=True)

        os.makedirs(os.path.join(
            self.q_src, self.session_id
        ), exist_ok=True)

        os.makedirs(os.path.join(
            self.q_stats
        ), exist_ok=True)
    
    def get_output_path_prefix(self):
        return os.path.join(
            self.q_out, self.session_id, self.query_hash, '{}'
        )
    
    def get_profiler_chunk(self):
        return os.path.join(
            self.dataset, 'chunked/chunk-13/relations'
        )

    def get_souffle_compiler_path(self):
        return self.souffle_comp

    def get_wrapper_path(self):
        return self.wrapper_sh
    
    def get_souffle_path(self):
        return self.souffle_sh
    
    def get_real_souffle_path(self):
        return self.souffle_comp.replace('souffle-compile', 'souffle')
    
    def get_prelude_prefix(self):
        return self.snippets_dir

    def get_prelude_file_path(self):
        return os.path.join(self.snippets_dir, 'prelude.dl')
    
    def get_utils_file_path(self):
        return os.path.join(self.snippets_dir, 'utils.dl')

    def get_prelude_file(self):
        with open(self.get_prelude_file_path(), 'r') as fh:
            return fh.read()

    def get_utils_file(self):
        with open(self.get_utils_file_path(), 'r') as fh:
            return fh.read()

    def get_query_hash(self):
        assert self.query_hash is not None, "Must call set_query_hash prior to get_query*"
        return self.query_hash

    def get_query_dl_file_path(self):
        assert self.query_hash is not None, "Must call set_query_hash prior to get_query*"
        return os.path.join(self.q_src, self.session_id,  self.query_hash + '.dl')
    
    def get_query_prof_dl_file_path(self):
        assert self.query_hash is not None, "Must call set_query_hash prior to get_query*"
        return os.path.join(self.q_src, self.session_id, self.query_hash + '.prof.dl')
    
    def get_query_prof_file_path(self):
        assert self.query_hash is not None, "Must call set_query_hash prior to get_query*"
        return os.path.join(self.q_src, self.session_id, self.query_hash + '.prof')
    
    def get_query_cpp_file_path(self, nosession=False):
        assert self.query_hash is not None, "Must call set_query_hash prior to get_query*"
        if nosession:
            return os.path.join(self.q_src, self.query_hash + '.cpp')
        return os.path.join(self.q_src, self.session_id, self.query_hash + '.cpp')
    
    def get_query_bin_file_path(self, nosession=False):
        assert self.query_hash is not None, "Must call set_query_hash prior to get_query*"
        if nosession:
            return os.path.join(self.q_src, self.query_hash)
        return os.path.join(self.q_src, self.session_id, self.query_hash)
    
    def get_query_stats_file_path(self, kind):
        assert self.query_hash is not None, "Must call set_query_hash prior to get_query*"
        return os.path.join(self.q_stats, self.query_hash + kind + '.log.txt')
    
    def write_query_dl(self, query):
        with open(self.get_query_dl_file_path(), 'w') as fh:
            fh.write(query)
    
    def write_query_prof(self, query_prof):
        with open(self.get_query_prof_dl_file_path(), 'w') as fh:
            fh.write(query_prof)
    
    def write_query_cpp(self, query):
        with open(self.get_query_cpp_file_path(), 'w') as fh:
            fh.write(query)

    def move_query_files_to_global_cache(self):
        globalize = lambda x: (x, x.replace(
            '/{}'.format(self.session_id), ''
        ))
        
        os.rename(*globalize(self.get_query_dl_file_path()))
        os.rename(*globalize(self.get_query_cpp_file_path()))
        os.rename(*globalize(self.get_query_prof_file_path()))
        os.rename(*globalize(self.get_query_bin_file_path()))

    def get_input_file_path(self, in_id):
        return os.path.join(
            self.q_in,
            self.session_id,
            self.query_hash,
            self.dataset.lstrip('/'),
            'input-{}.csv'.format(str(in_id))
        )    
    
    def get_query_is_compiled(self):
        if os.path.isfile(self.get_query_cpp_file_path(nosession=True)):
            return True
        return False

    def get_output_files(self, targets=None):
        if targets is not None:
            prefix = os.path.join(self.q_out, self.session_id, self.query_hash)
            return list(sorted([ p for p in [
                os.path.join(prefix, x.lstrip('/'), 'results/data.parquet') for x in targets
            ] if os.path.isfile(p) ]))

        return list(sorted([ p for p in glob.glob(
            os.path.join(
                self.q_out,
                self.session_id,
                self.query_hash,
                self.dataset.lstrip('/'),
                'chunked/chunk-*/relations/results/data.parquet'
            )
        ) if os.path.isfile(p) ]))


    

