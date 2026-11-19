/** How long a loaded model has been loaded and since its last request, as people say it ("loaded 3 h 5 min ago · last
 *  request 12 s ago"). The box sends seconds as of its answer (loaded_s_ago, idle_s): the page adds the time since it
 *  had the answer, so the ages run on between polls and no clock but the box's is trusted. */

/** A span of seconds: "12 s", "3 min", "2 h 14 min", "3 d 4 h" */
export function dur(s: number): string {
  const n = Math.max(0, Math.floor(s));
  if (n < 60) return `${n} s`;
  if (n < 3600) return `${Math.floor(n / 60)} min`;
  if (n < 86400) {
    const m = Math.floor((n % 3600) / 60);
    return `${Math.floor(n / 3600)} h${m ? ` ${m} min` : ""}`;
  }
  const h = Math.floor((n % 86400) / 3600);
  return `${Math.floor(n / 86400)} d${h ? ` ${h} h` : ""}`;
}

/** What a model (or one copy) says of its clocks */
export interface Clocks {
  loaded_s_ago?: number | null;
  idle_s?: number | null;
  unload_in_s?: number | null;
  busy?: boolean;
}

/** "loaded 2 h 14 min ago · last request 3 min ago" (or "· no request yet", "· in use now"); "" when the box does not
 *  say. `since`: seconds since the answer came. */
export function agesText(c: Clocks, since: number): string {
  if (c.loaded_s_ago == null) return "";
  const loaded = `loaded ${dur(c.loaded_s_ago + since)} ago`;
  if (c.idle_s == null) return `${loaded} · no request yet`;
  if (c.idle_s === 0 && c.busy) return `${loaded} · in use now`;
  return `${loaded} · last request ${dur(c.idle_s + since)} ago`;
}

/** "unloads in 7 min" when it will idle out; "" otherwise */
export function unloadText(c: Clocks, since: number): string {
  if (c.unload_in_s == null) return "";
  const left = c.unload_in_s - since;
  return left > 0 ? `unloads in ${dur(left)}` : "unloads now";
}
