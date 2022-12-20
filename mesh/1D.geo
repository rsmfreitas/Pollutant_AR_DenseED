// Gmsh project created on Mon Aug 29 11:52:31 2022
SetFactory("OpenCASCADE");
//+
lc = DefineNumber[ 2, Name "Parameters/lc" ];
//+
Point(1) = {0, 0, 0, lc};
//+
Point(2) = {100, 0, 0, lc};
//+
Point(3) = {100, 10, 0, lc};
//+
Point(4) = {0, 10, 0, lc};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Curve Loop(2) = {1, 2, 3, 4};
//+
Plane Surface(1) = {2};
//+
Physical Curve("left",5) = {4};
//+
Physical Curve("righ", 7) = {2};
//+
Physical Curve("bot", 6) = {1, 3};
//+
Physical Surface("surface", 8) = {1};
