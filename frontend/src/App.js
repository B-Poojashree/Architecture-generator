import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/common/Navbar";
import Footer from "./components/common/Footer";
import { AppProvider } from "./context/AppContext";

import Home from "./pages/Home/Home";
import Workflow from "./pages/Workflow/Workflow";
import RiskAnalysis from "./pages/RiskAnalysis/RiskAnalysis";
import TaskSchedule from "./pages/TaskSchedule/TaskSchedule";
import ModelComparison from "./pages/ModelComparison/ModelComparison";
import Results from "./pages/Results/Results";

function App() {
  return (
    <AppProvider>
      <BrowserRouter>
        <Navbar />
        <div className="container py-4">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/workflow" element={<Workflow />} />
            <Route path="/risk-analysis" element={<RiskAnalysis />} />
            <Route path="/task-schedule" element={<TaskSchedule />} />
            <Route path="/model-comparison" element={<ModelComparison />} />
            <Route path="/results" element={<Results />} />
          </Routes>
        </div>
        <Footer />
      </BrowserRouter>
    </AppProvider>
  );
}

export default App;
