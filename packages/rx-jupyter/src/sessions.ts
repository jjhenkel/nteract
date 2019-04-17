/**
 * @module rx-jupyter
 */
import { Observable } from "rxjs";
import { ajax, AjaxRequest, AjaxResponse } from "rxjs/ajax";
import { createAJAXSettings, ServerConfig } from "./base";

/**
 * Creates an AjaxObservable for listing available sessions.
 *
 * @param serverConfig The server configuration
 * @param sessionID Universally unique id for session to be requested.
 *
 * @returns An Observable with the request response
 */
export const list = (
  serverConfig: ServerConfig,
  opts: Partial<AjaxRequest & { cache?: boolean }> = {}
): Observable<AjaxResponse> =>
  ajax(
    createAJAXSettings(serverConfig, "/api/sessions", { cache: false, ...opts })
  );

/**
 * Creates an AjaxObservable for getting a particular session's information.
 *
 * @param serverConfig The server configuration
 * @param sessionID Universally unique id for session to be requested
 *
 * @returns An Observable with the request/response
 */
export const get = (
  serverConfig: ServerConfig,
  sessionID: string,
  opts: Partial<AjaxRequest & { cache?: boolean }> = {}
): Observable<AjaxResponse> =>
  ajax(
    createAJAXSettings(serverConfig, `/api/sessions/${sessionID}`, {
      cache: false,
      ...opts
    })
  );

/**
 * Creates an AjaxObservable for destroying a particular session.
 *
 * @param serverConfig The server configuration
 * @param sessionID Unique id for session to be requested
 *
 * @returns An Observable with the request/response
 */
export const destroy = (
  serverConfig: ServerConfig,
  sessionID: string,
  opts: Partial<AjaxRequest & { cache?: boolean }> = {}
): Observable<AjaxResponse> =>
  ajax(
    createAJAXSettings(serverConfig, `/api/sessions/${sessionID}`, {
      method: "DELETE",
      ...opts
    })
  );

/**
 * Creates an AjaxObservable for updating a session.
 *
 * @param serverConfig The server configuration
 * @param sessionID Unique identifier for session to be changed
 * @param body Payload containing new kernel_name, new kernel_id,
 * name of the new session, and the new path.
 *
 * @returns An Observable with the request/response
 */
export const update = (
  serverConfig: ServerConfig,
  sessionID: string,
  body: object,
  opts: Partial<AjaxRequest & { cache?: boolean }> = {}
): Observable<AjaxResponse> =>
  ajax(
    createAJAXSettings(serverConfig, `/api/sessions/${sessionID}`, {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json"
      },
      body,
      ...opts
    })
  );

/**
 * Creates an AjaxObservable for getting a particular session's information.
 *
 * @param serverConfig  The server configuration
 * @param body Payload containing kernel name, kernel_id, session
 * name, and path for creation of a new session.
 *
 * @returns An Observable with the request/response
 */
export const create = (
  serverConfig: ServerConfig,
  body: object,
  opts: Partial<AjaxRequest & { cache?: boolean }> = {}
): Observable<AjaxResponse> =>
  ajax(
    createAJAXSettings(serverConfig, "/api/sessions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body,
      ...opts
    })
  );
