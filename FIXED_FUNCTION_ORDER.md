# Fixed Function Order in ModelTraining Component

## Issues Fixed

1. **ReferenceError: Cannot access 'fetchTrainingHistory' before initialization**
   - Fixed the order of function declarations in the ModelTraining component
   - Moved useEffect after both fetch functions are defined
   - Ensured proper dependency management

## Technical Changes

### ModelTraining.jsx

1. **Reordered Function Declarations**
   - Moved the `fetchTrainingHistory` and `fetchPatients` function declarations before the `useEffect` that uses them:
     ```javascript
     // Define fetchTrainingHistory with useCallback
     const fetchTrainingHistory = useCallback(async () => {
       // function implementation
     }, []);

     const fetchPatients = useCallback(async () => {
       // function implementation
     }, []);

     // Add useEffect after both fetch functions are defined
     useEffect(() => {
       // Fetch training history
       fetchTrainingHistory();
       // Fetch patient list
       fetchPatients();
       // Set up polling interval
       // ...
     }, [fetchTrainingHistory, fetchPatients]);
     ```

2. **Proper Dependency Management**
   - Ensured that the `useEffect` dependency array includes both fetch functions:
     ```javascript
     useEffect(() => {
       // ...
     }, [fetchTrainingHistory, fetchPatients]);
     ```

## Why This Fixes the Issue

The error "Cannot access 'fetchTrainingHistory' before initialization" occurred because:

1. In JavaScript, variable declarations using `const`, `let`, or `var` are not hoisted with their initializations
2. When using `useCallback`, we're creating a function expression assigned to a variable, not a function declaration
3. The `useEffect` was trying to reference these functions in its dependency array before they were defined

By reordering the code to define the functions before using them in `useEffect`, we ensure that all references are valid when they're used.

## How to Test

1. Start the application using the run script:
   ```bash
   ./run_fixed_model.sh
   ```

2. Navigate to the Model Training page
3. Verify that the page loads without errors
4. Test the retraining functionality
5. Verify that the patient list updates correctly

The Model Training component should now work correctly without any reference errors.
