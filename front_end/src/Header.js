// import React from 'react';
//
// function Header() {
//   return (
//     <header style={{
//       backgroundColor: '#f8f9fa',
//       padding: '10px 20px',
//       borderBottom: '1px solid #dee2e6',
//       display: 'flex',
//       justifyContent: 'space-between',
//       alignItems: 'center'
//     }}>
//       <h1 style={{ margin: 0, fontSize: '24px', color: '#343a40' }}>SST</h1>
//     </header>
//   );
// }
//
// export default Header;

import React from 'react';
// import 'https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/minty/bootstrap.min.css';

const Header = () => {
    return (
        <nav className="navbar navbar-expand-lg navbar-dark bg-primary fixed-top">
            <div className="container-fluid">
                <a className="navbar-brand" href="#">2조프로젝트</a>
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
                            <a className="nav-link active" href="#">Home
                                <span className="visually-hidden">(current)</span>
                            </a>
                        </li>
                        <li className="nav-item">
                            <a className="nav-link" href="#">Transfer</a>
                        </li>

                        <li className="nav-item">
                            <a className="nav-link" href="#">About</a>
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