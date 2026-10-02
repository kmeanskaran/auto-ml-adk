// The little markdown agents write: headings, **bold**, `code` and pipe tables.
// Rendered as React elements, never as injected HTML.

import type { ReactNode } from "react";

export function Inline({ text }: { text: string }) {
  const parts: ReactNode[] = [];
  const pattern = /\*\*(.+?)\*\*|`([^`]+)`/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > last) parts.push(text.slice(last, match.index));
    parts.push(match[1] !== undefined ? <b key={match.index}>{match[1]}</b> : <code key={match.index}>{match[2]}</code>);
    last = match.index + match[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return <>{parts}</>;
}

const cells = (row: string) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());

export default function Markdown({ text }: { text: string }) {
  const lines = text.split("\n");
  const blocks: ReactNode[] = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^\s*\|/.test(line)) {
      const rows: string[] = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) {
        if (!/^\s*\|\s*:?-/.test(lines[i])) rows.push(lines[i]);
        i++;
      }
      i--;
      const [head, ...body] = rows;
      blocks.push(
        <table key={i}>
          <thead>
            <tr>
              {cells(head).map((c, j) => (
                <th key={j}>
                  <Inline text={c} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {body.map((r, n) => (
              <tr key={n}>
                {cells(r).map((c, j) => (
                  <td key={j}>
                    <Inline text={c} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>,
      );
    } else if (/^#{1,4}\s/.test(line)) {
      blocks.push(<h5 key={i}>{line.replace(/^#{1,4}\s+/, "")}</h5>);
    } else if (line.trim()) {
      blocks.push(
        <div key={i}>
          <Inline text={line} />
        </div>,
      );
    } else {
      blocks.push(<div key={i} className="gap" />);
    }
  }
  return <>{blocks}</>;
}
