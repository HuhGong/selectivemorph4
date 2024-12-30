import React from 'react';
import './App.css';
import Header from './Header';
import {BrowserRouter as Router, Routes, Route} from 'react-router-dom';
import PhotoSelector from "./PhotoSelector";
import Scroll from "./Scroll";
import About from './About';

function App() {
    return (
        <Router>
            <div>
                <Header/>
                <div className="main-container">
                    <Routes>
                        <Route path="/" element={<Scroll/>}/>
                        <Route path="/transfer" element={<PhotoSelector/>}/>
                        <Route path="/about" element={<About/>}/>
                        {/* 다른 경로에 대한 Route 추가 가능 */}
                    </Routes>
                </div>
            </div>
        </Router>
    );
}

export default App;
