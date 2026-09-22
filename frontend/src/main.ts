import "./style.css";
import { renderHome } from "./pages/home";
import { renderHistory } from "./pages/history";
import { renderSimulation } from "./pages/simulation";
import { renderPersons } from "./pages/persons";
import { renderSituationPage } from "./pages/situation";
import { startAlertWatcher } from "./watch";

type PageRenderer = (container: HTMLElement) => () => void;

const PAGES: Record<string, PageRenderer> = {
  home: renderHome,
  history: renderHistory,
  simulation: renderSimulation,
  persons: renderPersons,
};

const TABS: Array<[key: string, label: string]> = [
  ["home", "Home"],
  ["history", "History"],
  ["simulation", "Simulation"],
  ["persons", "People"],
];

let stopCurrentPage: (() => void) | null = null;

interface Route {
  page: string;
  situationId?: string;
}

function parseHash(): Route {
  const hash = location.hash.replace(/^#/, "");
  const situationMatch = hash.match(/^situation\/(.+)$/);
  if (situationMatch) return { page: "situation", situationId: decodeURIComponent(situationMatch[1]) };
  return { page: hash in PAGES ? hash : "home" };
}

function renderNav(active: string): void {
  const nav = document.getElementById("nav")!;
  nav.innerHTML = "";
  for (const [key, label] of TABS) {
    const link = document.createElement("a");
    link.href = `#${key}`;
    link.textContent = label;
    link.className = key === active ? "active" : "";
    nav.appendChild(link);
  }
}

function route(): void {
  stopCurrentPage?.();
  const { page, situationId } = parseHash();
  renderNav(page); // "situation" matches no tab, so none is highlighted -- it is not a top-level section
  const container = document.getElementById("page")!;
  container.innerHTML = "";
  stopCurrentPage = page === "situation" && situationId
    ? renderSituationPage(container, situationId)
    : PAGES[page](container);
}

window.addEventListener("hashchange", route);
route();

// Runs for the whole session, independent of the current page (DESIGN_DOCUMENT.md, Interface): "Upon receiving a
// warning or an alarm, the user should be notified and the view should switch to a page showing the situation."
startAlertWatcher((situationId) => {
  location.hash = `#situation/${encodeURIComponent(situationId)}`;
});
