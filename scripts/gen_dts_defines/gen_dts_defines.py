#!/usr/bin/env python3
'''
Utility to autogenerate generic LL headers with C-preprocessor macro defines.

Usage::

	python3 gen_dts_defines.py [-p /path/to/HAL] [-o /path/to/output_dir]

Copyright (c) 2025 Alexander Kozhinov <ak.alexander.kozhinov@gmail.com>

SPDX-License-Identifier: Apache-2.0
'''


import os
import re
import glob
import shutil
import logging
import datetime
import textwrap
import argparse
import tempfile
import threading
import itertools
import subprocess
from datetime import datetime


script_dir = os.path.dirname(os.path.abspath(__file__))  # this script directory


class gen_dts_defines:
	def __init__(self, stm32cube_path:str, dest_path: str, compiler:str=None,
				log_level:int=logging.INFO):
		app_name = 'gen_dts_defines'
		timestamp = ''  # datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
		log_file = os.path.join(script_dir, f'{app_name}_{timestamp}.log')
		self._lg = logging.getLogger(app_name)
		logging.basicConfig(
				filename=log_file,  # log file name
				level=log_level,  # minimum log level
				format='%(asctime)s: %(levelname)s: %(message)s'  # log format
			)

		self._stm32cube_path = stm32cube_path
		self._compiler = shutil.which('clang') if compiler == None else compiler
		self._dest_path = dest_path

		# List all series in stm32cube folder:
		stm32_series = [s for s in os.listdir(self._stm32cube_path) \
						if s.startswith('stm32')]
		stm32_series = [s[5:-2] for s in stm32_series]
		stm32_series = sorted(set(stm32_series))
		self._stm32_series = dict(zip(stm32_series, [[]]*len(stm32_series)))

		# For all series list SoCs:
		for s, _ in self._stm32_series.items():
			soc_dir = os.path.join(self._stm32cube_path, f'stm32{s}xx', 'soc')
			soc_list = [soc.split('_')[0][5:-2] for soc in os.listdir(soc_dir) \
						if soc.startswith('stm32')]
			soc_list = [soc.upper().replace('X','x') for soc in soc_list]  # make all characters capital exept x's
			soc_list.remove(s.upper() + 'xx')  # remove general series name
			soc_list = sorted(set(soc_list))
			self._stm32_series[s] = soc_list

		# Kep only u375xx and u385xx sub-series for u3 series:
		for s, _ in self._stm32_series.items():
			if s.lower() == 'u3':
				self._stm32_series[s] = [ss for ss in self._stm32_series[s] \
										if ss.lower() in ['u375xx', 'u385xx']]

		# For mp1, mp13 and mp2 series add 'xx' suffix to sub-series names:
		for s, _ in self._stm32_series.items():
			if s.lower() in ['mp1', 'mp13', 'mp2']:
				self._stm32_series[s] = [ss + 'xx' for ss in self._stm32_series[s]]

		# For l1 series last x charater shall be capitalized:
		for s, _ in self._stm32_series.items():
			if s.lower() == 'l1':
				self._stm32_series[s] = [ss[:-1] + 'X' if ss.endswith('x') else ss \
										for ss in self._stm32_series[s]]

		self._lg.info(f'Identified stm32 series:\n{self._stm32_series_pretty_str}')

		# Since this script extracts SoC peripherals defines from headers, there
		# is no interest in CPU headers used as a sub-dependency. Thus the CPU
		# headers may be leaved empty, but needed to be present in the include path.
		# Create dummy core_cXY.h empty headers:
		self._core_headers_tmp_dir = tempfile.mkdtemp(prefix='stm32_includes_')
		self._arm_cpu_names = [
			'core_cm0', 'core_cm0plus', 'core_cm3', 'core_cm4', 'core_cm7',
			'core_cm33', 'core_cm55', 'core_cm85',
			'core_ca']
		self._lg.info(f'Created folder {self._core_headers_tmp_dir}')
		for header in self._arm_cpu_names:
			header_path = os.path.join(self._core_headers_tmp_dir, header + '.h')
			with open(header_path, 'w', encoding='utf-8'): pass
			self._lg.info(f'Created dummy header {header_path}')

	def __del__(self):
		shutil.rmtree(self._core_headers_tmp_dir)
		self._lg.info(f'Removed folder: {self._core_headers_tmp_dir}')

	def _filter_defines_dump(self, def_dump_str: str, filter_prefix: str) -> dict:
		# Split string into lines by new-line symbol:
		def_dump = def_dump_str.split('\n')
		def_dump = [l.strip() for l in def_dump if l]
		def_dump = list(set(def_dump))  # unique entries

		# Remove #define keyword:
		start_str = '#define'
		def_dump = [l.split(start_str)[1].strip() for l in def_dump \
					if l.startswith(start_str)]

		# Filter out symbols starting with _ or __
		def_dump = [l for l in def_dump if not l.startswith('_')]

		# Build key-value pairs:
		split_idx_list = [l.find(' ') for l in def_dump]  # find indexes for first space occurence
		def_dump = {l[:idx].strip(): l[idx:].strip() \
					for l,idx in zip(def_dump, split_idx_list)}

		# Filter out function macro definitions:
		keys2keep = [k for k,_ in def_dump.items() if '(' not in k and ')' not in k]
		def_dump = {k: def_dump[k] for k in keys2keep}

		# Filter out entries with values starting with '_' character:
		keys2keep = [k for k,v in def_dump.items() \
					 if not v.replace('(','').startswith('_')]
		def_dump = {k: def_dump[k] for k in keys2keep}

		# Select keys by given prefix:
		fpr = filter_prefix.upper()
		selected_keys = set([k for k in def_dump.keys() if k.startswith(fpr)])

		# For each selected key recurisvely infer all dependent keys:

		checked_keys = set()
		while len(selected_keys - checked_keys):
			k = (selected_keys - checked_keys).pop()
			checked_keys.add(k)
			v = def_dump.get(k, None)
			if v == None:
				continue
			v_clean = self._split_val_by_opertor(v)
			selected_keys.update([dk for dk in def_dump.keys() if dk in v_clean])  # pick all keys in the value
		def_dump = {k: def_dump[k] for k in selected_keys}

		# For all numeric values in def dump drop non-numeric characters:
		for k,v in def_dump.items():
			v_clean = self._split_val_by_opertor(v)
			v_replace = v_clean.copy()
			v_replace = [vv if vv[0].isdigit() else vv for vv in v_replace]
			for i, vv in enumerate(v_replace):
				vv_lower = vv.lower()
				if vv_lower[0].isdigit():
					vv_prefix = ''
					if vv_lower.startswith('0x'):
						vv_prefix = '0x'
						vv_lower = vv_lower.replace(vv_prefix, '')
					vv_lower = ''.join(c for c in vv_lower if c.isdigit())
					vv_lower = vv_prefix + vv_lower
					v_replace[i] = vv_lower
			for i, (v_old, v_new) in enumerate(zip(v_clean, v_replace)):
				def_dump[k] = def_dump[k].replace(v_old, v_new)

		# Remove all spaces from values:
		for k,v in def_dump.items():
			def_dump[k] = v.replace(' ', '')

		# Remove type casts using regex
		for k,v in def_dump.items():
			def_dump[k] = re.sub(r'\(.*(\d+)?_t\)', '', v)

		def_dump = {k: def_dump[k] for k in sorted(def_dump)}  # sort by key
		return def_dump

	@staticmethod
	def _split_val_by_opertor(v):
		operators = ['(', ')', '+', '-', '*', '/', '%', '<<', '>>', '&', '|', '^']
		v_clean = v
		for op in operators: v_clean = v_clean.replace(op, ' ')
		v_clean = v_clean.split()
		return v_clean

	def _store_def(self, file_path: str, def_dump: str):
		if len(def_dump) == 0:
			self._lg.debug(f'No defines to store for file: {file_path}')
			return
		with open(file_path, 'w') as f:
			file_define_guard_str = f"{os.path.basename(file_path).upper().replace('.','_')}_"
			file_header = textwrap.dedent(f'''\
			/*
			 * NOTE: Autogenerated file using {os.path.basename(__file__)} script
			 *
			 * Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
			 * SPDX-License-Identifier: Apache-2.0
			 */

			#ifndef {file_define_guard_str}
			#define {file_define_guard_str}

			/* Peripheral defines */
			''')
			f.write(file_header)
			for k, v in def_dump.items(): f.write(f'#define {k} {v}\n');
			file_closure = f'\n#endif  /* {file_define_guard_str} */\n'
			f.write(file_closure)
		self._lg.debug(f'Created file: {file_path}')

	def _create_main_include(self, file_path: str, sub_series_list: list):
		file_define_guard_str = f"{os.path.basename(file_path).upper().replace('.','_')}_"
		sub_series_include_selector_str = ''
		sorted_sub_series_list = [s.lower() for s in sorted(sub_series_list)]
		for i,s in enumerate(sorted_sub_series_list):
			if i == 0:
				sub_series_include_selector_str += f'#if defined(stm32{s})\n'
			else:
				sub_series_include_selector_str += f'#elif defined(stm32{s})\n'
			sub_series_include_file =\
				f'{os.path.splitext(os.path.basename(file_path))[0].replace('_def',f'_stm32{s.lower()}_def')}.dtsi'
			sub_series_include_selector_str += f'  #include <st/h7/drivers/include/{sub_series_include_file}>\n'
		sub_series_include_selector_str += '#else\n  #error "Unknown series"\n#endif\n'
		file_content = textwrap.dedent(f'''
			/*
			 * NOTE: Autogenerated file using {os.path.basename(__file__)} script
			 *
			 * Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
			 * SPDX-License-Identifier: Apache-2.0
			 */

			#ifndef {file_define_guard_str}
			#define {file_define_guard_str}

			/* Include sub-series specific defines */
		''')
		file_content += sub_series_include_selector_str
		file_content += f'\n#endif  /* {file_define_guard_str} */\n'
		with open(file_path, 'w') as f:
			f.write(file_content)
		self._lg.debug(f'Created main include file: {file_path}')

	def _run_series_processing(self, series_name: str, sub_series_list: list, cpu_defines: list):
		series_dest_path = os.path.join(self._dest_path, series_name, 'drivers', 'include')
		os.makedirs(series_dest_path, exist_ok=True)
		self._lg.debug(f'Use series destination folder: {series_dest_path}')

		full_series_name = f'stm32{series_name}xx'
		series_include_path = os.path.join(self._stm32cube_path, full_series_name,
											'drivers','include')
		include_paths = [
				series_include_path,
				self._core_headers_tmp_dir,
				os.path.join(self._stm32cube_path, full_series_name, 'soc'),
			]
		include_paths = [['-I', ip] for ip in include_paths]
		include_paths = list(itertools.chain.from_iterable(include_paths))
		input_files_list = glob.glob(f'{series_include_path}/*_ll_*.h')
		self._lg.debug(f"Found headers:\n\t{'\n\t'.join(input_files_list)}")
		for input_file in input_files_list:
			self._lg.info(f'Processing file: {input_file}')
			result_file_name_prefix = os.path.splitext(os.path.basename(input_file))[0]
			for sub_series_name in sub_series_list:
				full_sub_series_name = f'STM32{sub_series_name}'
				self._lg.debug(f'Processing series: {full_sub_series_name}')
				defines = [
					f'-D{full_sub_series_name}',
				]
				if series_name.lower() == 'l1':
					defines += [f'-D{full_sub_series_name}']
				cmd = [self._compiler, '-dM', '-E', '-P', input_file] + \
					defines + cpu_defines + include_paths
				cmd_str = ' '.join(cmd)
				self._lg.debug(f'Execute command ({full_sub_series_name}): {cmd_str}')
				result = subprocess.run(cmd, capture_output=True, text=True)
				module_name = '_'.join(result_file_name_prefix.split('_')[-2:])  # pick last two parts separated by _ of the file name
				def_dump = self._filter_defines_dump(result.stdout, module_name)
				result_file_name = f'{result_file_name_prefix}_{full_sub_series_name.lower()}_def.dtsi'
				def_dump_dtsi = os.path.join(series_dest_path, result_file_name)
				self._store_def(def_dump_dtsi, def_dump)
				if len(result.stderr):
					self._lg.error(f'Command error ({full_sub_series_name}):\n{result.stderr}\nExecuted command: {cmd_str}\n')
					continue
				self._lg.debug(f'Processed series: {full_sub_series_name}')
			main_include_file = f'{result_file_name_prefix}_def.dtsi'
			# TODO: Not all sub-series have all the peripherals, so the main include
			# file may include non-existing files. Need to filter out non-existing
			# files.
			self._create_main_include(os.path.join(series_dest_path, main_include_file), sub_series_list)
			self._lg.info(f'Processed file: {input_file}\n')

	@property
	def _stm32_series_pretty_str(self) -> str:
		return '\n'.join([f'\t\'{k}\': {v}' for k,v in self._stm32_series.items()])

	def run(self):
		start_time = datetime.now()
		self._lg.info(f'Processing series:\n{self._stm32_series_pretty_str}')

		# Since only defines for peripherals matter we may safely add all available CPU defines
		cpu_defines = [f'-D{cpu_name.upper()}' for cpu_name in self._arm_cpu_names]
		threads = []
		# TODO: For performance reasons self._stm32_series shall be pandas table
		for series, series_list in self._stm32_series.items():
			# if series != 'mp1': continue;
			thread = threading.Thread(target=self._run_series_processing,
							args=(series, series_list, cpu_defines))
			threads.append(thread)
			thread.start()

		# Wait for all threads to complete
		for thread in threads:
			thread.join()
		elapsed_time = (datetime.now() - start_time).total_seconds()
		self._lg.info(f'Processing completed in {elapsed_time} sec.')



def main():
	envHOME = os.environ.get('HOME')
	stm32cube_path = os.path.abspath(f'{envHOME}/Documents/dev/zephyrproject/modules/hal/stm32/stm32cube')
	compiler = None  # os.path.abspath(f'{envHOME}/zephyr-sdk-0.17.2/arm-zephyr-eabi/bin/arm-zephyr-eabi-gcc')
	# compiler = '/usr/bin/cpp'
	dest_path = os.path.abspath(os.path.join(stm32cube_path,'..','dts','st'))

	gdd = gen_dts_defines(stm32cube_path, dest_path, compiler, log_level=logging.INFO)
	gdd.run()


if __name__ == '__main__':
	main()
