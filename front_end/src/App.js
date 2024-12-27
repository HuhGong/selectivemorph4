import React from 'react';
import './App.css';
import Header from './Header';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import PhotoSelector from "./PhotoSelector";
import Scroll from "./Scroll";
import AnnotationViewer from './AnnotationViewer'; // 추가된 컴포넌트 임포트
import Cover from './Cover';

function App() {
    return (
        <Router>
            <div>
                <Cover />
                {/*-<Header />*/}
                <div className="main-container">
                    <Routes>
                        <Route path="/" element={<Cover><Scroll /></Cover>} />
                        <Route path="/transfer" element={<Cover><PhotoSelector /></Cover>} />
                        {/* 다른 경로에 대한 Route 추가 가능 */}
                    </Routes>
                </div>
            </div>
        </Router>
    );
}

export default App;
