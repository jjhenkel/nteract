import pandas as pd
import pickle

import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

from codebook.python import *
from codebook.semantics.dsn import DSNotebooks as DSN

DATASET = '2020'

if DATASET == '2017':
  Evaluator.use_ds_gh_2017()
elif DATASET == '2019':
  Evaluator.use_ds_gh_2019()
elif DATASET == '2020':
  Evaluator.use_ds_gh_2020()
else:
  assert False, "Unknown dataset `{}`. (Valid options are: 2017/2019/2020)".format(DATASET)

print("SELECTED DATASET: `gh-{}`".format(DATASET))

for i in range(70, 100):
  try:
    print('Working on partition {}/{}'.format(i+1, 100))
    dsn = DSN((i, 100))

    results = dsn.run_all_operators()

    with open('/data/gh-{}/ip-results/results.part{}.pickle'.format(DATASET, str(i+1)), 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    all_ops = pd.concat([
      x['frame'][['fid', 'gid', 'pretty']] for x in results.values() if len(x['frame']) > 0 and x['ex_str'] == ''
    ])

    # Add "unmodeled" calls
    all_calls = dsn.exec(
      call() % select('name') % 'call'
    )

    # Get the ones we don't have models for 
    all_calls = all_calls[~all_calls.gid_call.isin(all_ops.gid)]
    all_calls['pretty'] = ('Unmodeled[' + all_calls.out_name_call + ']').astype('string')
    all_calls['gid'] = all_calls.gid_call.astype('Int64')

    # Add these now as "unmodeled" operators
    all_ops = pd.concat([all_ops, all_calls[['fid', 'gid', 'pretty']]])

    with open('/data/gh-{}/ip-results/all-ops.part{}.pickle'.format(DATASET, str(i+1)), 'wb') as handle:
        pickle.dump(all_ops, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Getting flows...')
    flows_df = dsn.flows_reads_to_fits()
    print('  + Got flows!')

    print('Joining ops and flows...')
    print('  + Step 1')
    grouped_ops = all_ops[['gid', 'pretty']].groupby(['gid']).agg(list).pretty.str.join(';')
    print('  + Step 2')
    flows_df = flows_df.set_index(
      'gid_flow', drop=False
    ).join(
      grouped_ops, rsuffix="_flow"
    ).set_index(
      'out_to_flow', drop=False
    ).join(
      grouped_ops, rsuffix="_out"
    )
    print('  + Joined!')

    with open('/data/gh-{}/ip-results/raw-flows.part{}.pickle'.format(DATASET, str(i+1)), 'wb') as handle:
        pickle.dump(flows_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('  + Grouping by....')
    implicit_flows = flows_df.sort_values(
      ['start_line_flow', 'start_col_flow']
    ).groupby(
      ['fid', 'gid_source', 'gid_sink']
    )[[
      'gid_flow', 'pretty', 
      'out_to_flow', 'pretty_out',
      'out_edge_flow'
    ]].agg(list)
    print('    + Grouped!')
    print('  + Post-processing done!')

    with open('/data/gh-{}/ip-results/processed-flows.part{}.pickle'.format(DATASET, str(i+1)), 'wb') as handle:
        pickle.dump(implicit_flows, handle, protocol=pickle.HIGHEST_PROTOCOL)

    graphs = []

    for key, row in implicit_flows.iterrows():
      tmp_g = nx.DiGraph()
      for ga, opa, gb, opb, edge in zip(*row.to_list()):
        tmp_g.add_node(ga, label=opa)
        tmp_g.add_node(gb, label=opb)
        tmp_g.add_edge(ga, gb, label=edge)
      graphs.append(tmp_g)

    with open('/data/gh-{}/ip-results/nx-graphs.part{}.pickle'.format(DATASET, str(i+1)), 'wb') as handle:
        pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('/data/gh-{}/ip-results/flows-graphs-debug.part{}.txt'.format(DATASET, str(i+1)), 'w') as fh:
      for i,g in enumerate(list(graphs)):
        print('{}/{}'.format(i+1, len(graphs)))
        fh.write(str(to_agraph(g)) + '\n---\n\n')

    del graphs
    del implicit_flows
    del grouped_ops
    del flows_df
    del all_ops
    del results
  except Exception as ex:
    print("Failed on partition {}/{}".format(i, 100))
    print(ex)
    print("  ! continuing...")
    continue

