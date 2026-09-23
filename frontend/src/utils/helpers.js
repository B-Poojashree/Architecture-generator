export function formatScore(value) {
  return typeof value === "number" ? value.toFixed(1) : value;
}

export function capitalize(text) {
  if (!text) return "";
  return text.charAt(0).toUpperCase() + text.slice(1);
}
