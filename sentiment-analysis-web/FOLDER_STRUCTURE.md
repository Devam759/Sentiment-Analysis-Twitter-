# Project Folder Structure

A comprehensive overview of the React + Vite Twitter Sentiment Analysis web application structure.

## 📁 Root Directory Structure

```
sentiment-analysis-web/
├── 📄 package.json                 # Project dependencies and scripts
├── 📄 vite.config.js              # Vite configuration
├── 📄 index.html                  # Main HTML file
├── 📄 README.md                   # Project documentation
├── 📄 FOLDER_STRUCTURE.md         # This file
├── 📄 start.bat / start.sh         # Startup scripts
├── 📁 src/                        # Source code
├── 📁 backend/                    # Flask API server
├── 📁 public/                     # Static assets
└── 📁 node_modules/               # Installed dependencies
```

## 📁 Source Code Structure (`src/`)

```
src/
├── 📄 App.jsx                     # Main React application component
├── 📄 App.css                     # Global styles
├── 📄 index.js                    # Application entry point
├── 📁 assets/                     # Static assets
│   ├── 📁 styles/                  # CSS and style files
│   └── 📁 images/                  # Image files
├── 📁 components/                 # Reusable React components
│   ├── 📁 common/                  # Common layout components
│   │   └── 📄 Navbar.jsx           # Navigation bar component
│   ├── 📁 ui/                       # UI components
│   │   ├── 📄 LoadingSpinner.jsx   # Loading spinner component
│   │   ├── 📄 ErrorAlert.jsx        # Error alert component
│   │   └── 📄 StatCard.jsx          # Statistics card component
│   ├── 📁 charts/                   # Chart components
│   │   ├── 📄 ModelPerformanceChart.jsx  # Model performance bar chart
│   │   └── 📄 SentimentDistributionChart.jsx # Sentiment doughnut chart
│   └── 📁 pages/                    # Page components
│       ├── 📄 Home.jsx              # Homepage component
│       ├── 📄 Analysis.jsx          # Analysis page component
│       └── 📄 About.jsx             # About page component
├── 📁 hooks/                      # Custom React hooks
│   ├── 📄 useSentimentAnalysis.js  # Sentiment analysis logic
│   ├── 📄 useModelComparison.js   # Model comparison logic
│   └── 📄 useStats.js              # Statistics logic
├── 📁 utils/                      # Utility functions
│   └── 📄 helpers.js              # Helper functions
├── 📁 constants/                  # Application constants
│   └── 📄 index.js                 # All constants in one file
└── 📁 api/                        # API service layer
    └── 📄 api.js                   # API service functions
```

## 📁 Backend Structure (`backend/`)

```
backend/
├── 📄 app.py                      # Flask API server
├── 📄 requirements.txt            # Python dependencies
└── 📁 __pycache__/                # Python cache files
```

## 📁 Component Categories

### 🎨 UI Components (`src/components/ui/`)
- **Purpose**: Reusable UI elements
- **Examples**: LoadingSpinner, ErrorAlert, StatCard
- **Characteristics**: No business logic, pure UI

### 📊 Chart Components (`src/components/charts/`)
- **Purpose**: Data visualization components
- **Examples**: ModelPerformanceChart, SentimentDistributionChart
- **Libraries**: Chart.js, react-chartjs-2

### 🧩 Common Components (`src/components/common/`)
- **Purpose**: Shared layout components
- **Examples**: Navbar, Footer, Header
- **Characteristics**: Used across multiple pages

### 📄 Page Components (`src/components/pages/`)
- **Purpose**: Full page components
- **Examples**: Home, Analysis, About
- **Characteristics**: Route-specific components

## 🎣 Custom Hooks (`src/hooks/`)

### useSentimentAnalysis.js
- Handles text prediction logic
- Manages prediction state and history
- Provides error handling and validation

### useModelComparison.js
- Manages model comparison data
- Handles multi-model analysis
- Provides model statistics

### useStats.js
- Fetches and manages application statistics
- Handles data formatting and processing
- Provides refresh functionality

## 🔧 Utilities (`src/utils/`)

### helpers.js
- Text formatting functions
- Validation functions
- Data transformation utilities
- Mock data generators

## 📋 Constants (`src/constants/`)

### index.js
- API endpoints
- Color definitions
- Configuration values
- Error messages
- Animation durations

## 🔌 API Layer (`src/api/`)

### api.js
- HTTP client configuration
- API endpoint functions
- Mock data fallbacks
- Error handling

## 🎨 Styling Approach

### CSS Organization
- **Global styles**: `App.css`
- **Component-specific**: Styled-components (Material-UI)
- **Theme configuration**: Material-UI theme in App.jsx

### Design System
- **Colors**: Consistent color palette in constants
- **Typography**: Material-UI typography system
- **Spacing**: Material-UI spacing system
- **Animations**: CSS transitions and Material-UI animations

## 🔄 Data Flow

```
User Input → Component → Custom Hook → API Service → Backend
                ↓
            State Update → UI Re-render
```

### Example Flow (Sentiment Prediction)
1. User types in SentimentPredictor component
2. Component calls useSentimentAnalysis hook
3. Hook validates input and calls API service
4. API service makes HTTP request to backend
5. Backend processes text with ML models
6. Result flows back through API service → hook → component
7. Component updates UI with prediction results

## 🏗️ Architecture Patterns

### Component Architecture
- **Container/Presentational**: Separation of logic and UI
- **Composition**: Small, reusable components
- **Props Drilling**: Minimal, using context when needed

### State Management
- **Local State**: useState for component-specific state
- **Custom Hooks**: Shared logic and state
- **No Global State**: Simple application doesn't need Redux/Zustand

### API Architecture
- **Service Layer**: Centralized API functions
- **Error Handling**: Consistent error handling across components
- **Mock Data**: Fallback for development without backend

## 📱 File Naming Conventions

### Components
- **PascalCase**: `ComponentName.jsx`
- **Descriptive**: `SentimentPredictor.jsx` not `Predictor.jsx`

### Hooks
- **use + PascalCase**: `useSentimentAnalysis.js`
- **Domain-specific**: Clear purpose in name

### Utilities
- **camelCase**: `helpers.js`, `constants.js`
- **Functional**: Describe what they do

## 🚀 Build Process

### Development
- **Vite Dev Server**: Fast hot reload
- **ESLint**: Code quality checks
- **Auto-imports**: Automatic import organization

### Production
- **Vite Build**: Optimized bundle
- **Code Splitting**: Automatic route-based splitting
- **Asset Optimization**: Image and CSS optimization

## 🔧 Configuration Files

### package.json
- Dependencies management
- Build scripts
- Development tools

### vite.config.js
- Build configuration
- Plugin setup
- Development server settings

### .env (optional)
- Environment variables
- API endpoints
- Feature flags

## 📦 Dependencies

### React Ecosystem
- **React**: UI library
- **React Router**: Client-side routing
- **Material-UI**: Component library

### Data & Charts
- **Chart.js**: Charting library
- **Axios**: HTTP client

### Development Tools
- **Vite**: Build tool
- **ESLint**: Code quality

This structure ensures:
✅ **Scalability**: Easy to add new features
✅ **Maintainability**: Clear separation of concerns
✅ **Reusability**: Modular components and hooks
✅ **Performance**: Optimized builds and lazy loading
✅ **Developer Experience**: Fast development and clear organization
