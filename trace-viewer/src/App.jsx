import { Routes, Route, BrowserRouter, useLocation } from 'react-router-dom';
import './App.css'
import TraceViewer from './TraceViewer';
import Home from './Home';

function AppContent() {
  const location = useLocation();
  // 检查是否有 trace 参数
  const urlParams = new URLSearchParams(location.search);
  const hasTrace = urlParams.has('trace');

  return hasTrace ? <TraceViewer /> : <Home />;
}

function App() {
  return (
    <BrowserRouter basename={process.env.NODE_ENV === 'production' ? '/spring2025-lectures/' : '/'}>
      <Routes>
        <Route path="/" element={<AppContent />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;