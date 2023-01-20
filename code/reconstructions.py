reconstructions = {
0 : """
First competition PB (38)
R' U' F R2 U L' D2 L2 U2 B F D2 F2 D2 R2 D B2 R D2 B U F2 R' U' F

(F') L2 U' R2 F2 // 2x2 (5/5)
U L U' L B2 L' B2 // Cross + 2. pair (7/12)
F R B U' B' R' F' // 3. pair (7/19)
U' L U2 L' // F2L (4/23)
F' U' F U' F' U [F U F R' F' R] // OLL (12/35)
[] = U F2 U F' U' F' L F L' // PLL (3-1/37)
U' // AUF (38)

L2 U' R2 F2 U L U' L B2 L' B2 F R B U' B' R' F' U' L U2 L' F' U' F U' F' U2 F2 U F' U' F' L F L' U' F'
""",
1 : """
Second competition PB (31)
R' U' F U2 B2 U2 F2 R2 D B2 U2 L2 F2 R B2 F U B' R' U L F' U2 R' U' F

B' D R2 F' U' F R2 U2 F2 U F // XX-Cross (11/11)
(B' U B) // 3. pair (3/14)
(L' U L) // F2L (3/17)
F' L2 B L B' L F // OLL to 3e (7-2/22)
[R2 U B' F R2 F' B U R2 U2] // (10/32)
[] = U2 L2 U F' B L2 F B' U L2 // Rewrite (0-1/31)

B' D R2 F' U' F R2 U2 F2 U L2 B L B' L F U2 L2 U F' B L2 F B' U L U' L B' U' B
""",
2 : """
Martin's no DR PB (25)
F U B2 R2 D F2 R2 D B2 D B2 U2 F' L2 U' F D' B L U2

(R2 U' L' B2 F U2) // Pseudo 2x2x3 (6/6)
(R D2 R' B2 D F) // 3c3e (6/12)
(R2 U' L' * B2 F U2 R D2 + R' B2 D F) // Skeleton
* = L' D2 L U L' D2 L U' // Inserted 3c (8-1/19)
+ = D L R' B2 R L' D // Inserted 3e (7-1/25)

F' D' B2 R D' L R' B2 R L' D R' U2 F' B2 U L' D2 L U' L' D2 L2 U R2
""",
3 : """
First completed DR solve, 12.11.22 (35)
B2 U2 R' B2 F2 U2 F2 R2 B2 R D2 F' D2 F' U F' U' R B L

L' B R' U2 F // EO (5/5)
D L' D L // Setup to 4c4e (4/9)
D' B2 U2 B2 D // DR trigger (5/14)
R' B2 L R' D2 * R2 B2 L2 // 2c2e (8/22)
* = D R U' R D' R2 U R' U' D R D' R2 U R'
// Inserted 2c2e (15-2/35)

L' B R' U2 F D L' D L D' B2 U2 B2 D R' B2 L R' D' R U' R D' R2 U R' U' D R D' R2 U R B2 L2
""",
4 : """
Second completed DR solve, 19.11.22 (30)
R' U' F L2 R2 U2 B2 D' R2 U' B2 L2 D2 L R2 B' F U R' B2 U' R2 U' R' U' R' U' F

(U' F L) // EO (3)
(F' * D F) // Setup to 4c4e (3/6)
(D F2 U' L2 D' U2 F) // DR (7/13)
(B2 U R2 +) // 3c2e2e (3/16)
* = F2 D B2 D' F2 D B2 D' // Inserted 3c (8-3/21)
+ = U F L2 F2 L2 F2 L2 F U' // Inserted 2e2e (9/30)

U F' L2 F2 L2 F2 L2 F' U' R2 U' B2 F' U2 D L2 U F2 D' F' B2 D' F2 D B2 D' F' L' F' U
""",
5 : """
Third completed DR solve, 01.12.22 (24)
R' U' F L2 D2 F L2 F U2 B L2 R2 B2 R2 U B' U' L' F2 U R' B' D' F R' U' F

L // EO (1/1)
B D2 B' // Setup to 4c4e (3/4)
D' U' B2 D' B // DR (5/9)
U2 L2 U F2 * L2 ** B2 // 3e3e (6/15)
* = E', ** = E // Inserted 3e (4/19)
L B D2 B' D' U' B2 D' B U2 L2 U F2 U + D' B2 U' D B2
// Skeleton
+ = F B' L2 F' B D2 // Inserted 3e (6-1/24)

L B D2 B' D' U' B2 D' B U2 L2 U F2 U F B' L2 F' B D B2 U' D B2
""",
6 : """
Fourth completed DR solve, 03.12.22 (29)
R' U' F L U2 B L2 F' L2 D2 U2 B L2 R2 B R' D' L D2 U' F2 U2 R' U' F

D' U' B // EO (3)
R' // Setup to 4c4e (1/4)
U F2 U L2 F2 U' L' // DR (7/11)
R2 D' + F2 D' L2 U2 D2 L2 * D' // 5e (9/20)
* = F B' L2 F' B D2 // Inserted 2e (6-1/25)
+ = D2 B F' R2 B' F // Inserted 3e (6-2/29)

D' U' B R' U F2 U L2 F2 U' L' R2 D B F' R2 B' F' D' L2 U2 D2 L2 F B' L2 F' B D
""",
7 : """
Fifth completed DR solve, 10.12.22 (23)
R' U' F L U' L D' R' B2 L' D' R D2 B D2 F B L2 F' R2 D2 R2 F R' U' F

R2 U' L' F B // EO (5/5)
R' // Setup to 4c4e (1/6)
U R2 U' L2 U' R // DR (6/12)
[* D B2 L2 F2 L2 + ** D'] // 3c3e (6/18)
* = E', ** = E // Inserted 3e (4-2/20)
+ = L2 U' R2 U L2 U' R2 U // Inserted 3c (8-3/25)
[] = U' D2 . L2 F2 R2 U' . B2 U . F2 U' . B2 D2 U2 .
..... = E2, E, E2, E, E2 // Rewrite (10-12/23)
[] = U R2 B2 L2 D' L2 U' D2 L2 D' F2

R2 U' L' F B R' U R2 U' L2 U' R U R2 B2 L2 D' L2 U' D2 L2 D' F2 }(23){
"""
}