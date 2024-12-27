import React from 'react';
import { Link } from 'react-router-dom';

const Header = () => {
    return (
        <nav className="navbar navbar-expand-lg navbar-dark bg-primary fixed-top"
             style={{ padding: '1.0rem', fontSize: '1.5rem' }}>
            <div className="container-fluid">
                <Link className="navbar-brand" to="/">2조프로젝트</Link>
                <button
                    className="navbar-toggler"
                    type="button"
                    data-bs-toggle="collapse"
                    data-bs-target="#navbarColor01"
                    aria-controls="navbarColor01"
                    aria-expanded="false"
                    aria-label="Toggle navigation"
                >
                    <span className="navbar-toggler-icon"></span>
                </button>
                <div className="collapse navbar-collapse" id="navbarColor01">
                    <ul className="navbar-nav me-auto">
                        <li className="nav-item">
                            <Link className="nav-link active" to="/">Home
                                <span className="visually-hidden">(current)</span>
                            </Link>
                        </li>
                        <li className="nav-item">
                            <Link className="nav-link" to="/transfer">Transfer</Link>
                        </li>
                        <li className="nav-item">
                            <Link className="nav-link" to="#">About</Link>
                        </li>
                    </ul>
                    <form className="d-flex">
                        <input className="form-control me-sm-2" type="search" placeholder="Search"/>
                        <button className="btn btn-secondary my-2 my-sm-0" type="submit">Search</button>
                    </form>
                </div>
            </div>
        </nav>
    );
};

export default Header;
