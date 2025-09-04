import { NavLink } from "react-router-dom";

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="navbar-container">
        <NavLink to="/" className="navbar-logo">
          Heart Failure Prediction System
        </NavLink>
        <div className="navbar-links">
          <NavLink
            to="/"
            className={({ isActive }) =>
              isActive ? "navbar-link active" : "navbar-link"
            }
          >
            New Patient
          </NavLink>
          <NavLink
            to="/history"
            className={({ isActive }) =>
              isActive ? "navbar-link active" : "navbar-link"
            }
          >
            Patient History
          </NavLink>
          <NavLink
            to="/retrain"
            className={({ isActive }) =>
              isActive ? "navbar-link active" : "navbar-link"
            }
          >
            Model Training
          </NavLink>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
