.. toctree::
   :maxdepth: 2
   :caption: Contents:


Coding Convention
~~~~~~~~~~~~~~~~~

Objectives of the Convention
****************************
**µSpectre** is a collaborative project and these coding conventions aim to make reading and understanding its code as pain-free as possible, while ensuring the four main requirements of the library
 #. Versatility
 #. Efficiency
 #. Reliability
 #. Ease-of-use

*Versatility* requires that the core of the code, i.e., the data structures and fundamental algorithms be written in a generic fashion. The genericity cannot come at the cost of the second requirement -- *Efficiency* -- which is the reason why the material base classes make extensive use of template metaprogramming and expression templates. *Reliability* can only be enforced through good unit testing with high test coverage, and *ease-of-use* relies on a good documentation for developers and users alike.

Review of submitted code is the main mechanism to enforce the coding conventions.

Structure
*********
.. image:: ../MainSchema.p

Documentation
*************
There are two types of Documentation for **µSpectre**: on the one hand, there is this monograph which is supposed to serve as reference manual to understand and use the library and its extensions, and to look up APIs and data structures, and on the other hand, there is in-code documentation helping the developer to understand the role of functions, variables, member (function)s and steps in algorithms

Testing
*******

Naming Conventions
******************

C++ Coding Style and Convention
*******************************

Python Coding Style
*******************
