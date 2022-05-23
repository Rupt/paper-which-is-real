.PHONY: examples
examples: example_ring_paper.log example_map_paper.log \
	example_ring.log example_map.log example_step.log


example_%.log: example_%.py sksym.py
	@mkdir -p example_$*
	python3 $< | tee $@


data_%.log: data_%.py
	@mkdir -p data_$*
	python3 $< | tee $@


# add special case prerequisites
example_map.log: data_map.log


.PHONY: fmt
fmt: *.py
	black *.py *.py -l79
	isort *.py *.py --profile black --line-length 79
	flake8  *.py; :


.PHONY: clean
clean:
	rm -rf example_*.log example_*/ data_*.log data_*/
