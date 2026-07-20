import re

with open('main.tex', 'r') as f:
    content = f.read()

tikz_code = r"""
\begin{figure*}[t]
\centering
\begin{tikzpicture}[
    node distance=1.5cm and 2cm,
    font=\sffamily\small,
    box/.style={draw, rounded corners, fill=blue!10, text width=2.5cm, align=center, minimum height=1.2cm},
    input/.style={draw, trapezium, trapezium left angle=70, trapezium right angle=-70, fill=green!10, align=center, minimum height=1cm},
    arrow/.style={-stealth, thick},
    groupbox/.style={draw, dashed, inner sep=10pt, rounded corners}
]

% Inputs
\node[input] (dfsar) {DFSAR (L/S)\\ CPR \& DOP};
\node[input, right=1cm of dfsar] (lola) {LOLA DEM\\ \& Slope};
\node[input, right=1cm of lola] (ephem) {Orbital\\ Ephemeris};

% Detector Stages
\node[box, below=1.5cm of lola, fill=orange!15] (stage1) {Stage 1: Sunlit\\ Reference Extraction};
\node[box, below=0.8cm of stage1, fill=orange!15] (stage2) {Stage 2: Anomaly\\ Scoring ($z_{CPR}, z_{DOP}$)};
\node[box, below=0.8cm of stage2, fill=orange!15] (stage3) {Stage 3: Roughness\\ Veto ($\le 2\sigma$)};
\node[box, below=0.8cm of stage3, fill=orange!15] (stage4) {Stage 4: Spatial\\ Coherence (DBSCAN)};
\node[box, below=0.8cm of stage4, fill=orange!15] (stage5) {Stage 5: Dual-Freq\\ Corroboration};

% Bounding box for detector
\node[groupbox, fit=(stage1) (stage5), label={[shift={(0,-4.5)}]\textbf{5-Stage AND-Cascade}}] (cascade) {};

% Maps Output
\node[box, fill=cyan!15, left=1cm of stage5] (pi_map) {Ice Probability\\ Map ($\Pi$)};
\node[box, fill=red!15, right=1cm of stage5] (phi_map) {False Positive\\ Risk Map ($\Phi$)};

% Planner
\node[box, fill=purple!15, below=1.5cm of stage5, text width=4cm] (planner) {A* Traverse Planner\\ $C(p) = f(\Pi, \Phi, \text{Slope})$};
\node[box, fill=yellow!20, below=1cm of planner, text width=3cm] (output) {Optimal Rover\\ Traverse Path};

% Routing Arrows
\draw[arrow] (dfsar.south) -- ++(0,-0.5) -| (stage1.north);
\draw[arrow] (lola.south) -- (stage1.north);
\draw[arrow] (ephem.south) -- ++(0,-0.5) -| (stage1.north);

\draw[arrow] (stage1) -- (stage2);
\draw[arrow] (stage2) -- (stage3);
\draw[arrow] (stage3) -- (stage4);
\draw[arrow] (stage4) -- (stage5);

\draw[arrow] (stage5.west) -- (pi_map.east);
\draw[arrow] (stage5.east) -- (phi_map.west);

\draw[arrow] (pi_map.south) |- (planner.west);
\draw[arrow] (phi_map.south) |- (planner.east);
\draw[arrow] (lola.east) -- ++(4.5,0) |- (planner.east);
\draw[arrow] (ephem.east) -- ++(2.5,0) |- (planner.east);

\draw[arrow] (planner) -- (output);

\end{tikzpicture}
\caption{End-to-End System Architecture. The multi-modal orbital data (DFSAR, LOLA, Ephemeris) feeds into the 5-stage AND-cascade detector. The cascade outputs high-level Ice Probability ($\Pi$) and False Positive Risk ($\Phi$) maps, which are jointly optimized with slope and orbital relay constraints in the A* Traverse Planner to output physically feasible rover sorties.}
\label{fig:architecture}
\end{figure*}
"""

# Insert right after \section{Proposed Framework}
target = r"\section{Proposed Framework}" + "\n" + r"\label{sec:method}" + "\n"
if target in content:
    content = content.replace(target, target + "\n" + tikz_code)
    with open('main.tex', 'w') as f:
        f.write(content)
    print("Successfully inserted TikZ Flowchart")
else:
    print("Could not find Proposed Framework section")
