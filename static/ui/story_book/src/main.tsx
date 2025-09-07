import React from "react";
import ReactDOM from "react-dom/client";

import "./styles.css";
import { Home } from "./ui/Home";

const rootElement = document.getElementById("root")!;
if (!rootElement.innerHTML) {
  const root = ReactDOM.createRoot(rootElement);
  root.render(
    <React.StrictMode>
      <Home />
    </React.StrictMode>,
  );
}
