########################################################################
#                          University of Costa Rica
#                      Electrical Engineering Department
#                                Grade Thesis
# ---------------------------------------------------------------------
#
# File: Makefile
# Author: Pablo Avila [B30724]
# Email: jose.avilalopez@ucr.ac.cr
# Copyright: Copyright 2020, LFP Classification
# License: MIT License
# ---------------------------------------------------------------------
#
# Description:
# This makefile generates the source code documentation using doxygen
# and pdflatex.
########################################################################

Target = refman.pdf
LatexFolder = latex

all: documentation pdf back

documentation: 
	doxygen Doxyfile

pdf:
	cd latex; make 

back:
	cd ..


view:
	evince $(LatexFolder)/$(Target)
clean:
	rm -rf \latex \html *.ps *.dvi *.aux *.toc *.idx *.ind *.ilg *.log *.out *.brf *.blg *.bbl $(Target) 
