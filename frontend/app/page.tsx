'use client';

import { useState } from "react";
import axios from 'axios';
import { Input } from "@nextui-org/input";
import { Button } from "@nextui-org/button";

export default function Home() {
  const [formData, setFormData] = useState({
    age: '',
    gender: '',
    cigsPerDay: '',
    totChol: '',
    sysBP: '',
    glucose: '',
  });
  
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Handle input change
  const handleChange = (e: { target: { name: any; value: any; }; }) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  // Handle form submission
  const handleSubmit = async (e: { preventDefault: () => void; }) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    
    try {
      const config = {
        method: 'post',
        url: 'http://127.0.0.1:5000/predict',
        headers: { 'Content-Type': 'application/json' },
        data: JSON.stringify(formData),
      };
      
      const response = await axios.request(config);
      setResult(response.data);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <section className="flex flex-col items-center justify-center py-8 md:py-10 ">
      <div className="p-8 rounded-lg shadow-md w-full max-w-md">
        <h2 className="text-2xl  font-semibold text-center mb-6">Heart Disease Prediction</h2>
        
        <form onSubmit={handleSubmit} className="flex flex-col gap-6">
          <Input 
            type="number" 
            name="age" 
            label="Age" 
            placeholder="Enter your age"  
            value={formData.age} 
            onChange={handleChange} 
            required 
            className="w-full"
          />
          <Input 
            type="number" 
            name="gender" 
            label="Gender (1 for male, 0 for female)" 
            placeholder="Enter gender"  
            value={formData.gender} 
            onChange={handleChange} 
            required 
            className="w-full"
          />
          <Input 
            type="number" 
            name="cigsPerDay" 
            label="Cigarettes per Day" 
            placeholder="Enter cigarettes per day"  
            value={formData.cigsPerDay} 
            onChange={handleChange} 
            required 
            className="w-full"
          />
          <Input 
            type="number" 
            name="totChol" 
            label="Total Cholesterol" 
            placeholder="Enter total cholesterol"  
            value={formData.totChol} 
            onChange={handleChange} 
            required 
            className="w-full"
          />
          <Input 
            type="number" 
            name="sysBP" 
            label="Systolic Blood Pressure" 
            placeholder="Enter systolic blood pressure"  
            value={formData.sysBP} 
            onChange={handleChange} 
            required 
            className="w-full"
          />
          <Input 
            type="number" 
            name="glucose" 
            label="Glucose Level" 
            placeholder="Enter glucose level"  
            value={formData.glucose} 
            onChange={handleChange} 
            required 
            className="w-full"
          />

          <Button color="primary" variant="bordered" type="submit" className="mt-4">
            Predict
          </Button>
        </form>

        {result && (
          <div className="mt-6 p-4 bg-green-100 text-green-800 rounded-lg">
            <h3 className="font-medium">Prediction Result</h3>
            <p>{JSON.stringify(result)}</p>
          </div>
        )}

        {error && (
          <div className="mt-6 p-4 bg-red-100 text-red-800 rounded-lg">
            <p>{error}</p>
          </div>
        )}
      </div>
    </section>
  );
}
