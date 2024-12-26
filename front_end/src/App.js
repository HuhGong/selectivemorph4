import React from 'react';
import './App.css';
import Header from './Header';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import PhotoSelector from "./PhotoSelector";
import Scroll from "./Scroll";
import AnnotationViewer from './AnnotationViewer'; // 추가된 컴포넌트 임포트

function App() {
    return (
        <Router>
            <div>
                <Header />
                <div className="main-container">
                    <div className="scroll-container">
                        <Scroll />
                    </div>
                    <div className="photo-selector-container">
                        <PhotoSelector />
                    </div>
                </div>
            </div>
        </Router>
    );
}

export default App;
