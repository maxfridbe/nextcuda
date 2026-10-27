/** The "non-commercial" tag of a model whose licence forbids commercial use (its model card's `commercial: false`),
 *  with the licence and its conditions as the tooltip (the same in every app). */
import { jsx } from "../jsx.js";

export interface Licence { license?: string | null; commercial?: boolean | null; license_note?: string | null }

export const NonCommercial = (m: Licence | null | undefined) =>
  m && m.commercial === false
    ? <span class="tag nc" attrs={{ title: `${m.license ?? "licence"}: no commercial use${m.license_note ? ` - ${m.license_note}` : ""}` }}>non-commercial</span>
    : null;
