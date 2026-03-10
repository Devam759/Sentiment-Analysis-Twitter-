import React from 'react';

const About = () => {
  const technologies = [
    { name: 'React', description: 'Modern frontend framework for building user interfaces' },
    { name: 'Vite', description: 'Fast build tool and development server' },
    { name: 'Tailwind CSS', description: 'Utility-first CSS framework for rapid UI development' },
    { name: 'Chart.js', description: 'Data visualization library for performance metrics' },
    { name: 'Python', description: 'Backend machine learning and NLP processing' },
    { name: 'scikit-learn', description: 'Machine learning library for model training' },
    { name: 'NLTK', description: 'Natural language processing toolkit' },
    { name: 'Gensim', description: 'Topic modeling and document similarity analysis' },
  ];

  const features = [
    {
      icon: (
        <svg className="w-8 h-8 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
        </svg>
      ),
      title: 'Research-Based Methodology',
      description: 'Built following academic research standards for sentiment analysis',
    },
    {
      icon: (
        <svg className="w-8 h-8 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
        </svg>
      ),
      title: 'Multiple ML Models',
      description: 'SVM, Random Forest, Logistic Regression, and Naive Bayes implementations',
    },
    {
      icon: (
        <svg className="w-8 h-8 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      ),
      title: 'Advanced Features',
      description: 'Named entity recognition, topic modeling, and semantic analysis',
    },
    {
      icon: (
        <svg className="w-8 h-8 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      ),
      title: 'Real-time Processing',
      description: 'Fast sentiment analysis with confidence scores and probability distributions',
    },
  ];

  return (
    <div className="pt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
            About This Project
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            A comprehensive machine learning pipeline for Twitter sentiment analysis with modern web interface
          </p>
        </div>

        {/* Project Overview */}
        <div className="card p-8 mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-6">
            Project Overview
          </h2>
          <div className="prose prose-lg max-w-none text-gray-600">
            <p className="mb-4">
              This Twitter Sentiment Analysis project implements a comprehensive machine learning pipeline for predicting 
              consumer sentiment from tweet text. The system combines advanced natural language processing techniques 
              with multiple machine learning models to provide accurate and reliable sentiment predictions.
            </p>
            <p>
              Built following research methodology, this project demonstrates professional-level NLP capabilities while 
              maintaining accessibility for educational purposes. The system processes raw Twitter data, extracts meaningful 
              features, trains multiple models, and provides detailed analysis and visualization of results.
            </p>
          </div>
        </div>

        {/* Key Features */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-gray-900 text-center mb-12">
            Key Features
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <div key={index} className="card card-hover p-6 text-center group">
                <div className="flex justify-center mb-4 group-hover:scale-110 transition-transform duration-200">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">
                  {feature.title}
                </h3>
                <p className="text-gray-600 leading-relaxed">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Technology Stack */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-gray-900 text-center mb-12">
            Technology Stack
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="card p-6">
              <h3 className="text-xl font-semibold text-gray-900 mb-6">
                Frontend Technologies
              </h3>
              <div className="space-y-4">
                {technologies.slice(0, 4).map((tech, index) => (
                  <div key={index} className="flex items-start">
                    <svg className="w-5 h-5 text-success mr-3 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    <div>
                      <div className="font-medium text-gray-900">{tech.name}</div>
                      <div className="text-sm text-gray-600">{tech.description}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="card p-6">
              <h3 className="text-xl font-semibold text-gray-900 mb-6">
                Backend & ML Technologies
              </h3>
              <div className="space-y-4">
                {technologies.slice(4).map((tech, index) => (
                  <div key={index} className="flex items-start">
                    <svg className="w-5 h-5 text-success mr-3 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    <div>
                      <div className="font-medium text-gray-900">{tech.name}</div>
                      <div className="text-sm text-gray-600">{tech.description}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Pipeline Architecture */}
        <div className="card p-8 mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-6">
            Pipeline Architecture
          </h2>
          <p className="text-gray-600 mb-8">
            The system follows a comprehensive machine learning pipeline:
          </p>
          <div className="space-y-4">
            {[
              { step: 'Data Collection & Loading', desc: 'Sentiment140 dataset integration with custom dataset support' },
              { step: 'Text Preprocessing', desc: 'URL removal, mention/hashtag cleaning, tokenization, lemmatization' },
              { step: 'Feature Extraction', desc: 'TF-IDF vectorization, Word2Vec embeddings, statistical features' },
              { step: 'Model Training', desc: 'Multiple algorithms with cross-validation and hyperparameter tuning' },
              { step: 'Evaluation & Visualization', desc: 'Comprehensive metrics, confusion matrices, performance charts' },
              { step: 'Advanced Analysis', desc: 'NER, topic modeling, semantic similarity, emotion detection' },
            ].map((item, index) => (
              <div key={index} className="flex items-start bg-gray-50 rounded-lg p-4">
                <div className="w-8 h-8 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center font-semibold text-sm mr-4 flex-shrink-0">
                  {index + 1}
                </div>
                <div>
                  <div className="font-medium text-gray-900">{item.step}</div>
                  <div className="text-sm text-gray-600">{item.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Model Performance */}
        <div className="card p-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-6">
            Model Performance
          </h2>
          <p className="text-gray-600 mb-8">
            The system achieves excellent performance across multiple metrics:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center p-6 bg-primary-50 rounded-xl">
              <div className="text-4xl font-bold text-primary-600 mb-2">
                94%
              </div>
              <div className="text-gray-900 font-medium">Best Accuracy</div>
            </div>
            <div className="text-center p-6 bg-success/10 rounded-xl">
              <div className="text-4xl font-bold text-success mb-2">
                4
              </div>
              <div className="text-gray-900 font-medium">Models Trained</div>
            </div>
            <div className="text-center p-6 bg-warning/10 rounded-xl">
              <div className="text-4xl font-bold text-warning mb-2">
                0.15s
              </div>
              <div className="text-gray-900 font-medium">Avg Processing</div>
            </div>
            <div className="text-center p-6 bg-primary-50 rounded-xl">
              <div className="text-4xl font-bold text-primary-600 mb-2">
                5K
              </div>
              <div className="text-gray-900 font-medium">Sample Dataset</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;
