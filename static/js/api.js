export async function apiRequest(path, options = {}) {
  const { allowUnauthorized = false, ...fetchOptions } = options;
  const response = await fetch(path, {
    credentials: "include",
    ...fetchOptions,
    headers: {
      ...(fetchOptions.body && !(fetchOptions.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...fetchOptions.headers,
    },
  });
  if (response.status === 401 && !allowUnauthorized) {
    window.dispatchEvent(new CustomEvent("fancart:unauthorized"));
  }
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error(`The server returned an invalid response (${response.status}).`);
  }
  if (!response.ok) {
    const error = new Error(payload.error || `Request failed (${response.status}).`);
    error.status = response.status;
    throw error;
  }
  return payload;
}

export function postJSON(path, value, options = {}) {
  return apiRequest(path, {
    ...options,
    method: "POST",
    body: JSON.stringify(value),
  });
}
