// Vendor modules
import * as monaco from "monaco-editor";
import {
  actions,
  AppState,
  createContentRef,
  createHostRef,
  createKernelRef,
  createKernelspecsRef,
  makeAppRecord,
  makeCommsRecord,
  makeContentsRecord,
  makeDummyContentRecord,
  makeEntitiesRecord,
  makeHostsRecord,
  makeJupyterHostRecord,
  makeStateRecord,
  makeTransformsRecord,
} from "@nteract/core";
import { Media } from "@nteract/outputs";
import TransformVDOM from "@nteract/transform-vdom";
import { ContentRecord, HostRecord } from "@nteract/types";
import * as Immutable from "immutable";
import * as React from "react";
import ReactDOM from "react-dom";
import { Provider } from "react-redux";

// Local modules
import App from "./app";
import { JupyterConfigData } from "./config";
import("./fonts");
import configureStore from "./store";

export async function main(
  config: JupyterConfigData,
  rootEl: Element
): Promise<void> {
  // When the data element isn't there, provide an error message
  // Primarily for development usage

  const jupyterHostRecord = makeJupyterHostRecord({
    id: null,
    type: "jupyter",
    defaultKernelName: "python",
    token: config.token,
    origin: location.origin,
    basePath: config.baseUrl,
    bookstoreEnabled: config.bookstore.enabled,
    showHeaderEditor: false,
  });

  const hostRef = createHostRef();
  const contentRef = createContentRef();
  const NullTransform = () => null;

  monaco.languages.registerHoverProvider('*', {
    provideHover: (model, position) => {
      let allMatches = []
      for (let k = 0; k < (window as any).hoverTips.length; k++) {
        let res = (window as any).hoverTips[k](position);
        if (res) {
          allMatches.push(res);
        }
      }
      if (allMatches.length > 0) {
        return {
          range: allMatches[0].range,
          contents: allMatches.map(m => m.contents).flat()
        };
      }
    },
  });

  const omit = (obj: any, omitKey: string) => {
    return Object.keys(obj).reduce((result, key) => {
      if(key !== omitKey) {
         result[key] = obj[key];
      }
      return result;
    }, {});
  };
  
  const ShowDotTransform = (data: any) => {
    console.log('In: ShowDotTransform');
    (window as any).sidePanelResults = [{
      hasCaret: false,
      isExpanded: true,
      label: 'Graph (Dot)',
      id: 1,
      nodeData: { onClick: () => null },
      childNodes: []
    }];

    (window as any).sidePanelLanguage = data.data['lang'];
    (window as any).sidePanelDot = data.data['dot'];

    return null;
  };
  
  const PrintTransform = (data: any) => {
    console.log('In: PrintTransform');
    console.log((data.data['results'] || []).length);
    (window as any).sidePanelResults = [];
    (window as any).sidePanelLanguage = data.data['lang'];
    let resCount = 0;
    let tid = 0;
    let results = data.data['results'] || [];

    if (results !== []) {
      (window as any).sidePanelDot = null;
    }

    for (let i = 0; i < results.length; i++) {
      if (results[i]['$match'] && results[i]['$match'].length > 0) {
        for (let j = 0; j < results[i]['$match'].length; j++) {
          if (resCount > 100) {
            break;
          }
          resCount += 1;

          let gid = results[i]['$match'][j]['gid'];
          let fpath = results[i]['$match'][j]['fpath'];

          let onClick = (node: any, e: any) => {

            // Deselect
            let worklist = [];
            (window as any).sidePanelResults.forEach(x => worklist.push(x));
            while (worklist.length > 0) {
              let item = worklist.pop();
              item.isSelected = false;
              (item.childNodes || []).forEach(x => worklist.push(x));
            }
            
            node.isSelected = true;

            (window as any).store.dispatch(
              actions.fetchContent({
                filepath: fpath.replace('/cb-target/', '/processed/'),
                params: {},
                kernelRef: createKernelRef(),
                contentRef: 'e169379a-32ce-452d-b821-f76f8d61dd2d'
              })
            );

            let allDecorations = [];
            (window as any).hoverTips = [];

            Object.keys(results[i]).forEach((key) => {

              let sl = results[i][key][j]['s_line'] + 1 || 0;
              let sc = results[i][key][j]['s_col'] || 0;
              let el = results[i][key][j]['e_line'] + 1 || 0;
              let ec = results[i][key][j]['e_col'] + 1 || 0;
              
              allDecorations.push(
                  { range: new monaco.Range(sl,sc,el,ec), options: { 
                    linesDecorationsClassName: 'cbSelectionMarker',
                    inlineClassName: 'cbInlineHighlight1' 
                  }}
              );

              (window as any).hoverTips.push((pos) => {
                if (pos.lineNumber >= sl && pos.column >= sc && pos.lineNumber <= el && pos.column <= ec) {
                  return {
                    range: new monaco.Range(sl,0,el,100),
                    contents: [
                      { value: '***' + key + '***' },
                      { value: '```json\n' + JSON.stringify(omit(results[i][key][j], 'text'), null, 2) + '\n```' }
                    ]
                  };
                }
                return null;
              });

              if (key === '$match') {
                (window as any).sideEditor.revealLineInCenter(
                  Math.floor((sl + el) / 2)
                );
              }
            });

            (window as any).sidePanelSelect = () => {
              (window as any).sideEditor.deltaDecorations([], allDecorations);
            };

          };

          let pathParts = fpath.split('/').filter(x => x !== '');
          let current = (window as any).sidePanelResults;

          pathParts.forEach((p) => {
            tid += 1;
            let match = current.findIndex(e => e.label === p);
            if (match === -1) {
              current.push({
                hasCaret: true,
                isExpanded: true,
                label: p,
                id: tid,
                nodeData: { onClick: () => null },
                childNodes: []
              });
              current = current[current.length - 1].childNodes;
            } else {
              current = current[match].childNodes;
            }
          });

          tid += 1;
          current.push({
            id: tid,
            hasCaret: false,
            nodeData: { onClick },
            label: 'Match #' + (current.length + 1).toString()
          });
        }
      }

      if ((window as any).sidePanelResults.length > 10000) {
        console.log("Warning: too many results to visualize.")
        break;
      }
    }

    let worklist = [];
    
    (window as any).sidePanelResults.forEach(node => worklist.push(node));
    
    while (worklist.length > 0) {
      let item = worklist.pop();
      
      if (!item.childNodes) {
        continue;
      }

      while (item.childNodes.length === 1) {
        if (item.childNodes[0].label.startsWith('Match #') || item.childNodes[0].label.includes('.')) {
          break;
        }
        item.label += '/' + (
          item.childNodes[0].label
        );
        item.childNodes = (
          item.childNodes[0].childNodes
        );
      }

      item.childNodes.forEach(node => worklist.push(node));
    }

    return null;
  };

  const kernelspecsRef = createKernelspecsRef();

  const initialState: AppState = {
    app: makeAppRecord({
      version: `nteract-on-jupyter@${config.appVersion}`,
      host: jupyterHostRecord,
    }),
    comms: makeCommsRecord(),
    config: Immutable.Map({
      theme: "light",
    }),
    core: makeStateRecord({
      currentKernelspecsRef: kernelspecsRef,
      entities: makeEntitiesRecord({
        hosts: makeHostsRecord({
          byRef: Immutable.Map<string, HostRecord>().set(
            hostRef,
            jupyterHostRecord
          ),
        }),
        contents: makeContentsRecord({
          byRef: Immutable.Map<string, ContentRecord>().set(
            contentRef,
            makeDummyContentRecord({
              filepath: config.contentsPath,
            })
          ),
        }),
        transforms: makeTransformsRecord({
          displayOrder: Immutable.List([
            "application/code-book-matches+json",
            "application/code-book-dot+json",
            "application/vnd.jupyter.widget-view+json",
            "application/vnd.vega.v5+json",
            "application/vnd.vega.v4+json",
            "application/vnd.vega.v3+json",
            "application/vnd.vega.v2+json",
            "application/vnd.vegalite.v3+json",
            "application/vnd.vegalite.v2+json",
            "application/vnd.vegalite.v1+json",
            "application/geo+json",
            "application/vnd.plotly.v1+json",
            "text/vnd.plotly.v1+html",
            "application/x-nteract-model-debug+json",
            "application/vnd.dataresource+json",
            "application/vdom.v1+json",
            "application/json",
            "application/javascript",
            "text/html",
            "text/markdown",
            "text/latex",
            "image/svg+xml",
            "image/gif",
            "image/png",
            "image/jpeg",
            "text/plain",
          ]),
          byId: Immutable.Map({
            "application/code-book-matches+json": PrintTransform,
            "application/code-book-dot+json": ShowDotTransform,
            "text/vnd.plotly.v1+html": NullTransform,
            "application/vnd.plotly.v1+json": NullTransform,
            "application/geo+json": NullTransform,
            "application/x-nteract-model-debug+json": NullTransform,
            "application/vnd.dataresource+json": NullTransform,
            "application/vnd.jupyter.widget-view+json": NullTransform,
            "application/vnd.vegalite.v1+json": NullTransform,
            "application/vnd.vegalite.v2+json": NullTransform,
            "application/vnd.vegalite.v3+json": NullTransform,
            "application/vnd.vega.v2+json": NullTransform,
            "application/vnd.vega.v3+json": NullTransform,
            "application/vnd.vega.v4+json": NullTransform,
            "application/vnd.vega.v5+json": NullTransform,
            "application/vdom.v1+json": TransformVDOM,
            "application/json": Media.Json,
            "application/javascript": Media.JavaScript,
            "text/html": Media.HTML,
            "text/markdown": Media.Markdown,
            "text/latex": Media.LaTeX,
            "image/svg+xml": Media.SVG,
            "image/gif": Media.Image,
            "image/png": Media.Image,
            "image/jpeg": Media.Image,
            "text/plain": Media.Plain,
          }),
        }),
      }),
    }),
  };

  const kernelRef = createKernelRef();

  const store = configureStore(initialState);
  (window as any).store = store;

  store.dispatch(
    actions.fetchContent({
      filepath: config.contentsPath,
      params: {},
      kernelRef,
      contentRef,
    })
  );
  store.dispatch(actions.fetchKernelspecs({ hostRef, kernelspecsRef }));

  ReactDOM.render(
    <React.Fragment>
      <Provider store={store}>
        <App contentRef={contentRef} />
      </Provider>
    </React.Fragment>,
    rootEl
  );
}
