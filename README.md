# Just a quick note
---
1. The `main.py` is where we will conduct the tests.
2. The `match.py` is the main function to use ORFLANSAC, EST-LAU ORB, and modified versions of those algorithms (through parameter changes).
    - Imports of needed processes and algorithms are done here.
3. Note that there are bits of code that must be deleted (or archived) after tests are done. Possible files are:
    - `match.py`
    - `ofs.py`
4. The `ofs.py` is utilized by `match.py`. It is where the main processes of the algorithm lives. These processes or algorithms include:
    - image pre-processing 
    - Bayesian Optimization
    - ORB
    - MAGSAC++
    - specialized conditions (for TESTING)
5. DO NOT USE IT YET (under development): The `window.py` is where our GUI for simulation lives. 
    - It could also be re-used and modified for the GUI of the PRODUCT SEARCH project.

# Changes (Nov. 17, 2024)
---
1. Created a separate `orflansac.py` file. 
2. Corrected test cases.
3. Deleted the images of `creamo` because it is not part of the obtained dataset from Roboflow.
4. Updated this README.md file
