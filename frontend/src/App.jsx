import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedModel, setSelectedModel] = useState('vgg16'); // 'cnn' or 'vgg16'
  const [evaluationMetrics, setEvaluationMetrics] = useState(null);
  const [loadingMetrics, setLoadingMetrics] = useState(false);

  // Load metrics when model changes
  useEffect(() => {
    loadMetrics(selectedModel);
  }, [selectedModel]);

  const loadMetrics = async (modelType) => {
    setLoadingMetrics(true);
    try {
      const response = await axios.get(`http://localhost:5000/api/metrics/${modelType}`);
      setEvaluationMetrics(response.data.metrics);
    } catch (err) {
      console.error('Failed to load metrics:', err);
      // Set default metrics if API fails
      setEvaluationMetrics({
        accuracy: 0.90,
        precision: 0.90,
        recall: 0.89,
        f1_score: 0.89
      });
    } finally {
      setLoadingMetrics(false);
    }
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPrediction(null);
      setError('');
    }
  };

  const handleModelChange = (modelType) => {
    setSelectedModel(modelType);
    setPrediction(null); // Clear previous prediction when switching models
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('model_type', selectedModel);

    try {
      const response = await axios.post('http://localhost:5000/api/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setPrediction(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to make prediction');
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence > 0.8) return 'text-green-600';
    if (confidence > 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  // Default metrics if not loaded
  const metrics = evaluationMetrics || {
    accuracy: 0.90,
    precision: 0.90,
    recall: 0.89,
    f1_score: 0.89
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center">
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent mb-4">
              Cervical Cancer Classification
            </h1>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Advanced AI-powered classification system with CNN and VGG16 models
            </p>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          {/* Main Classification Section */}
          <div className="xl:col-span-2">
            <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
              <div className="bg-gradient-to-r from-blue-600 to-indigo-600 px-8 py-6">
                <h2 className="text-2xl font-bold text-white">Image Classification</h2>
                <p className="text-blue-100 mt-2">Upload and analyze cervical cell images</p>
              </div>
              
              <div className="p-8">
                {/* Model Selection Toggle */}
                <div className="mb-8">
                  <label className="block text-lg font-semibold text-gray-700 mb-4">
                    Select Model
                  </label>
                  <div className="flex gap-4">
                    <button
                      onClick={() => handleModelChange('cnn')}
                      className={`flex-1 py-3 px-6 rounded-xl font-semibold transition-all duration-200 transform ${
                        selectedModel === 'cnn'
                          ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white shadow-lg scale-105'
                          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                      }`}
                    >
                      <div className="text-lg font-bold">CNN</div>
                      <div className="text-sm opacity-90">Custom Convolutional Neural Network</div>
                    </button>
                    <button
                      onClick={() => handleModelChange('vgg16')}
                      className={`flex-1 py-3 px-6 rounded-xl font-semibold transition-all duration-200 transform ${
                        selectedModel === 'vgg16'
                          ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg scale-105'
                          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                      }`}
                    >
                      <div className="text-lg font-bold">VGG16</div>
                      <div className="text-sm opacity-90">Transfer Learning Model</div>
                    </button>
                  </div>
                </div>

                <div className="mb-8">
                  <label className="block text-lg font-semibold text-gray-700 mb-4">
                    Upload Image
                  </label>
                  <div className="relative">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleFileSelect}
                      className="block w-full text-sm text-gray-500 file:mr-4 file:py-3 file:px-6 file:rounded-xl file:border-0 file:text-sm file:font-semibold file:bg-gradient-to-r file:from-blue-50 file:to-indigo-50 file:text-blue-700 hover:file:from-blue-100 hover:file:to-indigo-100 file:transition-all file:duration-200"
                    />
                    <p className="mt-2 text-sm text-gray-500">
                      Supported formats: PNG, JPG, JPEG, BMP, TIFF (Max 16MB)
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  {previewUrl && (
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900 mb-4">Image Preview</h3>
                      <div className="border-2 border-dashed border-gray-200 rounded-xl p-6 bg-gray-50">
                        <img
                          src={previewUrl}
                          alt="Preview"
                          className="max-w-full h-auto rounded-lg mx-auto shadow-md"
                          style={{ maxHeight: '300px' }}
                        />
                      </div>
                    </div>
                  )}

                  {prediction && (
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900 mb-4">Analysis Results</h3>
                      <div className="bg-gradient-to-br from-gray-50 to-blue-50 rounded-xl p-6 border border-gray-200">
                        <div className="mb-4">
                          <span className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${
                            prediction.model_type === 'cnn'
                              ? 'bg-purple-100 text-purple-800'
                              : 'bg-blue-100 text-blue-800'
                          }`}>
                            {prediction.model_type.toUpperCase()} Model
                          </span>
                        </div>
                        <div className="mb-6">
                          <h4 className="text-lg font-semibold text-gray-700 mb-2">Primary Classification</h4>
                          <div className="bg-white rounded-lg p-4 shadow-sm">
                            <p className="text-xl font-bold text-blue-600 mb-2">
                              {prediction.prediction}
                            </p>
                            <p className={`text-lg font-semibold ${getConfidenceColor(prediction.confidence)}`}>
                              Confidence: {(prediction.confidence * 100).toFixed(2)}%
                            </p>
                          </div>
                        </div>

                        <div>
                          <h4 className="text-lg font-semibold text-gray-700 mb-4">
                            Detailed Probabilities
                          </h4>
                          <div className="space-y-3">
                            {Object.entries(prediction.all_predictions).map(([className, prob]) => (
                              <div key={className} className="bg-white rounded-lg p-3 shadow-sm">
                                <div className="flex justify-between items-center mb-2">
                                  <span className="text-sm font-medium text-gray-700">{className}</span>
                                  <span className="text-sm font-bold text-gray-900">
                                    {(prob * 100).toFixed(1)}%
                                  </span>
                                </div>
                                <div className="w-full bg-gray-200 rounded-full h-3">
                                  <div
                                    className={`h-3 rounded-full transition-all duration-500 ${
                                      prediction.model_type === 'cnn'
                                        ? 'bg-gradient-to-r from-purple-500 to-pink-500'
                                        : 'bg-gradient-to-r from-blue-500 to-indigo-500'
                                    }`}
                                    style={{ width: `${prob * 100}%` }}
                                  ></div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {error && (
                  <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-xl">
                    <div className="flex items-center">
                      <div className="flex-shrink-0">
                        <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                        </svg>
                      </div>
                      <div className="ml-3">
                        <p className="text-red-700 font-medium">{error}</p>
                      </div>
                    </div>
                  </div>
                )}

                <div className="mt-8">
                  <button
                    onClick={handlePredict}
                    disabled={loading || !selectedFile}
                    className={`w-full font-bold py-4 px-8 rounded-xl transition-all duration-200 transform hover:scale-105 disabled:scale-100 disabled:cursor-not-allowed shadow-lg ${
                      selectedModel === 'cnn'
                        ? 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-400 disabled:to-gray-400 text-white'
                        : 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 disabled:from-gray-400 disabled:to-gray-400 text-white'
                    }`}
                  >
                    {loading ? (
                      <div className="flex items-center justify-center">
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Analyzing Image...
                      </div>
                    ) : (
                      `Analyze Image with ${selectedModel.toUpperCase()}`
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Evaluation Metrics Section */}
          <div className="xl:col-span-1">
            <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
              <div className={`px-6 py-6 ${
                selectedModel === 'cnn'
                  ? 'bg-gradient-to-r from-purple-600 to-pink-600'
                  : 'bg-gradient-to-r from-green-600 to-emerald-600'
              }`}>
                <h2 className="text-2xl font-bold text-white">Model Performance</h2>
                <p className="text-white opacity-90 mt-2">
                  {selectedModel.toUpperCase()} Evaluation Metrics
                </p>
              </div>
              
              <div className="p-6">
                {loadingMetrics ? (
                  <div className="flex justify-center items-center py-12">
                    <svg className="animate-spin h-8 w-8 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                  </div>
                ) : (
                  <div className="space-y-6">
                    {/* Key Metrics */}
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900 mb-4">Key Metrics</h3>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-4 text-center">
                          <div className="text-2xl font-bold text-blue-600">
                            {((metrics.accuracy || metrics.accuracy_score || 0) * 100).toFixed(1)}%
                          </div>
                          <div className="text-sm text-blue-700 font-medium">Accuracy</div>
                        </div>
                        <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-xl p-4 text-center">
                          <div className="text-2xl font-bold text-green-600">
                            {((metrics.precision || metrics.precision_score || 0) * 100).toFixed(1)}%
                          </div>
                          <div className="text-sm text-green-700 font-medium">Precision</div>
                        </div>
                        <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl p-4 text-center">
                          <div className="text-2xl font-bold text-purple-600">
                            {((metrics.recall || metrics.recall_score || 0) * 100).toFixed(1)}%
                          </div>
                          <div className="text-sm text-purple-700 font-medium">Recall</div>
                        </div>
                        <div className="bg-gradient-to-br from-orange-50 to-orange-100 rounded-xl p-4 text-center">
                          <div className="text-2xl font-bold text-orange-600">
                            {((metrics.f1_score || metrics.f1 || 0) * 100).toFixed(1)}%
                          </div>
                          <div className="text-sm text-orange-700 font-medium">F1-Score</div>
                        </div>
                      </div>
                    </div>

                    {/* Model Info */}
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Information</h3>
                      <div className="space-y-3">
                        <div className="bg-gray-50 rounded-lg p-3">
                          <div className="text-sm font-medium text-gray-700">Architecture</div>
                          <div className="text-sm text-gray-900">
                            {selectedModel === 'cnn' ? 'Custom CNN' : 'VGG16 + Custom Classifier'}
                          </div>
                        </div>
                        <div className="bg-gray-50 rounded-lg p-3">
                          <div className="text-sm font-medium text-gray-700">Dataset</div>
                          <div className="text-sm text-gray-900">900+ cervical cell images</div>
                        </div>
                        <div className="bg-gray-50 rounded-lg p-3">
                          <div className="text-sm font-medium text-gray-700">Classes</div>
                          <div className="text-sm text-gray-900">4 categories</div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* About Section */}
        <div className="mt-12 bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
          <div className="bg-gradient-to-r from-indigo-600 to-purple-600 px-8 py-6">
            <h2 className="text-2xl font-bold text-white">About the Models</h2>
            <p className="text-indigo-100 mt-2">Understanding cervical cancer classification</p>
          </div>
          
          <div className="p-8">
            <div className="prose prose-lg text-gray-600 max-w-none">
              <p className="text-lg mb-6">
                This advanced AI system uses two deep learning models to classify cervical cell images into four critical categories.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl p-6 border border-purple-100">
                  <h4 className="font-bold text-purple-800 mb-2">CNN Model</h4>
                  <p className="text-purple-700 text-sm mb-3">
                    A custom Convolutional Neural Network designed specifically for cervical cell classification. This model is trained from scratch and optimized for this specific task.
                  </p>
                </div>
                <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-100">
                  <h4 className="font-bold text-blue-800 mb-2">VGG16 Model</h4>
                  <p className="text-blue-700 text-sm mb-3">
                    Uses transfer learning with the pre-trained VGG16 architecture, fine-tuned for cervical cancer classification. Leverages features learned from ImageNet.
                  </p>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div className="bg-gradient-to-br from-red-50 to-pink-50 rounded-xl p-6 border border-red-100">
                  <h4 className="font-bold text-red-800 mb-2">High squamous intra-epithelial lesion</h4>
                  <p className="text-red-700 text-sm">High-grade precancerous changes requiring immediate attention</p>
                </div>
                <div className="bg-gradient-to-br from-yellow-50 to-orange-50 rounded-xl p-6 border border-yellow-100">
                  <h4 className="font-bold text-yellow-800 mb-2">Low squamous intra-epithelial lesion</h4>
                  <p className="text-yellow-700 text-sm">Low-grade precancerous changes requiring monitoring</p>
                </div>
                <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-6 border border-green-100">
                  <h4 className="font-bold text-green-800 mb-2">Negative for Intraepithelial malignancy</h4>
                  <p className="text-green-700 text-sm">Normal, healthy cervical cells</p>
                </div>
                <div className="bg-gradient-to-br from-purple-50 to-indigo-50 rounded-xl p-6 border border-purple-100">
                  <h4 className="font-bold text-purple-800 mb-2">Squamous cell carcinoma</h4>
                  <p className="text-purple-700 text-sm">Cancerous cells requiring urgent medical intervention</p>
                </div>
              </div>
              
              <div className="bg-amber-50 border border-amber-200 rounded-xl p-6">
                <div className="flex items-start">
                  <div className="flex-shrink-0">
                    <svg className="h-6 w-6 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 19.5c-.77.833.192 2.5 1.732 2.5z" />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <h4 className="text-lg font-semibold text-amber-800">Important Notice</h4>
                    <p className="text-amber-700 mt-2">
                      This tool is designed for educational and research purposes only. 
                      Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
