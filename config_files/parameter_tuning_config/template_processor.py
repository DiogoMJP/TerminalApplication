from string import Template

for range in [1, 2, 3]:
	for mutate_power in [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3]:
		with open(f"r_{range}_mp_{mutate_power}_config", "w+") as fp:
			with open("config_template", "r") as src_fp:
				src = Template(src_fp.read())
			pattern = {
				"range"	: range,
				"stdev"	: round(range / 3, 2),
				"mutate_power"	: mutate_power,
				"num_inputs"	: "$num_inputs",
				"num_outputs"	: "$num_outputs"
			}
			fp.write(src.substitute(pattern))