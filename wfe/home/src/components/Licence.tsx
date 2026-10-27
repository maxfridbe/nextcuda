/** A model's licence tag (the same in every app), with the licence and its conditions as the tooltip:
 *  "non-commercial" when it forbids commercial use (its card's `commercial: false`), "conditional licence" when it
 *  allows commercial use only on conditions (`commercial: true` with a `license_note`: territories, user or revenue
 *  limits, a component under another licence), nothing for a plain licence or one not known. */
import { jsx } from "../jsx.js";

export interface Licence { license?: string | null; commercial?: boolean | null; license_note?: string | null }

export const NonCommercial = (m: Licence | null | undefined) =>
  !m ? null
    : m.commercial === false
      ? <span class="tag nc" attrs={{ title: `${m.license ?? "licence"}: no commercial use${m.license_note ? ` - ${m.license_note}` : ""}` }}>non-commercial</span>
      : m.commercial === true && m.license_note
        ? <span class="tag cl" attrs={{ title: `${m.license ?? "licence"}: commercial use on conditions - ${m.license_note}` }}>conditional licence</span>
        : null;
