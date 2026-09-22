const VISIBLE_MS = 8000;

/** A transient toast in the top-right corner, for "the user should be notified" -- no permission prompt needed
 * (unlike the browser Notification API), so it always works. */
export function showToast(message: string, kind: "warning" | "alarm" = "warning"): void {
  const stack = document.getElementById("toasts");
  if (!stack) return;

  const toast = document.createElement("div");
  toast.className = `toast ${kind}`;
  toast.textContent = message;
  stack.appendChild(toast);

  requestAnimationFrame(() => toast.classList.add("show"));
  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => toast.remove(), 300);
  }, VISIBLE_MS);
}
