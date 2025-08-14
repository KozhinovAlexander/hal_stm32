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
import pandas as pd
from datetime import datetime


script_dir = os.path.dirname(os.path.abspath(__file__))  # this script directory

# Configure pandas to show full data-frames when printed:
pd.set_option('display.max_columns', None)  # show all columns
pd.set_option('display.max_rows', None)     # show all rows
pd.set_option('display.width', None)        # don't wrap wide tables
pd.set_option('display.max_colwidth', None) # show full content of each column


class gen_dts_defines:
	def __repr__(self):
		return f'{self.__class__.__name__}(stm32cube_path=\'{self._stm32cube_path}\', ' \
			   f'dest_path=\'{self._dest_path}\', series={self._series}, ' \
			   f'compiler=\'{self._compiler}\')'

	def __init__(self, stm32cube_path:str, dest_path: str, compiler:str=None,
				series:list[str]=[], log_level:int=logging.INFO):
		app_name = self.__class__.__name__
		timestamp = ''  # f"_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
		log_file = os.path.join(script_dir, f'{app_name}{timestamp}.log')
		self._lg = logging.getLogger(app_name)
		logging.basicConfig(
				filename=log_file,  # log file name
				level=log_level,  # minimum log level
				format='%(asctime)s: %(threadName)s: %(levelname)s: %(message)s'  # log format
			)

		self._stm32cube_path = stm32cube_path
		self._compiler = compiler
		self._dest_path = dest_path
		self._series = series

		# List all series in stm32cube folder:
		stm32_series = [s for s in os.listdir(self._stm32cube_path) \
						if s.startswith('stm32')]
		stm32_series = [s[5:-2] for s in stm32_series]
		stm32_series = sorted(set(stm32_series))

		# For all series list SoCs and add them to data-frame:
		self._stm32_series_df = pd.DataFrame(columns=['series','soc'])
		for s in stm32_series:
			soc_dir = os.path.join(self._stm32cube_path, f'stm32{s}xx', 'soc')
			soc_list = [soc.split('_')[0][5:-2] for soc in os.listdir(soc_dir) \
						if soc.startswith('stm32')]
			soc_list = [soc.upper().replace('X','x') for soc in soc_list]  # make all characters capital exept x's
			soc_list.remove(s.upper() + 'xx')  # remove general series name
			soc_list = sorted(set(soc_list))
			for soc in soc_list:
				self._stm32_series_df.loc[len(self._stm32_series_df)] =\
					{'series': s, 'soc': soc}
		self._stm32_series_df.sort_values('series', inplace=True)
		self._stm32_series_df.reset_index(drop=True, inplace=True)

		# Keep only u375xx and u385xx soc-names for u3 series in dataframe:
		self._stm32_series_df = self._stm32_series_df[
			~((self._stm32_series_df['series'].str.lower() == 'u3') & \
			  (~self._stm32_series_df['soc'].str.lower().isin(['u375xx', 'u385xx'])))]
		self._stm32_series_df.reset_index(drop=True, inplace=True)

		# For mp1, mp13 and mp2 series add 'xx' suffix to soc values in dataframe:
		mp_series = ['mp1', 'mp13', 'mp2']
		for s in mp_series:
			self._stm32_series_df.loc[self._stm32_series_df['series'].str.lower() == s, 'soc'] += 'xx'

		# For l1 series last x charater shall be capitalized in dataframe:
		self._stm32_series_df.loc[self._stm32_series_df['series'].str.lower() == 'l1', 'soc'] = \
			self._stm32_series_df.loc[self._stm32_series_df['series'].str.lower() == 'l1', 'soc'].apply(
				lambda x: x[:-1] + 'X' if x.endswith('x') else x)

		self._core_headers_tmp_dir = None

		self._stm32_series_df = self._stm32_series_df.drop_duplicates(subset=['soc'])
		self._stm32_series_df.sort_values('series', inplace=True)
		self._stm32_series_df.reset_index(drop=True, inplace=True)

	def __del__(self):
		if self._core_headers_tmp_dir != None:
			if os.path.exists(self._core_headers_tmp_dir):
				shutil.rmtree(self._core_headers_tmp_dir)
				self._lg.info(f'Removed folder: {self._core_headers_tmp_dir}')

	@staticmethod
	def _filter_defines_dump(logger: logging.Logger, def_dump_str: str) -> pd.DataFrame:
		def_dump_list = def_dump_str.split('\n')
		not_visit_idx:int = -1
		def_df = pd.DataFrame(list(zip([not_visit_idx] * len(def_dump_list),
			[None] * len(def_dump_list), [None] * len(def_dump_list),
			[None] * len(def_dump_list), [None] * len(def_dump_list), def_dump_list)),
			columns=['visit-idx', 'key', 'value', 'val-split-op', 'val-split-op-replacement', 'line'])

		def_df = def_df[def_df['line'] != '']  # drop lines with an empty string
		def_df[['key', 'value']] = def_df['line'].apply(lambda l:
				pd.Series(l.strip()[len('#define '):].split(' ',1)))  # form value and key columns

		def_df = def_df[~def_df['key'].str.startswith('_')]  # filter out keys starting with _
		def_df = def_df[~def_df['key'].str.contains(r'\(|\)')]  # filter out function-like macros
		# def_df.drop('line', axis=1, inplace=True)  # drop line column (optional since make execution slightly slower)
		def_df.sort_values('key', inplace=True)

		def_df = def_df[def_df['value'].notna()]  # drop all rows, where value is undefined
		def_df['value'] = def_df['value'].apply(lambda v: v.replace(' ', ''))  # remove spaces
		def_df['value'] = def_df['value'].apply(lambda v: re.sub(r'\w+\d+_t', '', v))  # remove types of form int8_t
		def_df['value'] = def_df['value'].apply(lambda v: re.sub(r'\(\)', '', v))  # remove parentheses with empty content ()

		# Select keys by given prefix:
		fpr = 'LL_'  # prefix to filter defines for
		visit_idx:int = 0
		def_df['visit-idx'] = def_df['key'].apply(
			lambda k: visit_idx if k.startswith(fpr) else not_visit_idx)

		# Visit each sub-key while counting visit steps:
		keys2visit = [None]*2  # provide list with fake length for the start
		while len(keys2visit) > 0:
			# Split value by operator and drop starting with digit:
			def_df['val-split-op'] = def_df.loc[def_df['visit-idx'] == visit_idx, 'value']\
				.apply(lambda v: [k for k in gen_dts_defines._split_val_by_opertor(v) \
									if not k[0].isdigit()])

			# Determine keys to be visited:
			keys2visit = sorted(set(itertools.chain.from_iterable(def_df['val-split-op'].dropna())))
			visit_idx += 1

			# Mark keys2visit by new visit_idx:
			def_df.loc[def_df['key'].isin(keys2visit), 'visit-idx'] = visit_idx
		def_df = def_df[def_df['visit-idx'] != not_visit_idx]  # select all visited rows

		# For all numeric characters in value column remove non-digit values in numbers:
		def_df['val-split-op'] = def_df['value'].apply(
			lambda v: gen_dts_defines._split_val_by_opertor(v))

		def_df['val-split-op-replacement'] = def_df['val-split-op'].apply(
			lambda val_list: [gen_dts_defines._format_numeric(v) for v in val_list])

		# In value column replace all elements from val-split-op with their replacements:
		if len(def_df) != 0:
			def_df['value'] = def_df.apply(lambda row:\
				gen_dts_defines._replace_in_str(row['value'], row['val-split-op'],
												row['val-split-op-replacement']), axis=1)

		# Overwrite line column with cleaned values:
		def_df['line'] = def_df.apply(lambda row: f"#define {row['key']} {row['value']}\n", axis=1)

		# Sort and reindex:
		def_df.sort_values('key', inplace=True)
		def_df.reset_index(drop=True, inplace=True)  # reindex in place

		# Use following two lines for debug purposes only:
		# logger.info(f"\n{def_df[['visit-idx', 'key', 'value', 'val-split-op', 'val-split-op-replacement', 'line']]}")
		# assert False, '---> STOP <---'

		return def_df

	@staticmethod
	def _replace_in_str(s: str, old: list[str], new: list[str]) -> str:
			if len(old) != len(new): return s;
			if old == new: return s;
			sv = s
			for o,n in zip(old, new): sv = sv.replace(o, n);
			return sv

	@staticmethod
	def _format_numeric(v:str) -> str:
		v_ret = v
		if v[0].isdigit():
			if v[:2].lower() == '0x':
				# For hex values, keep the 0x prefix and only the hex digits
				v_ret = '0x' + ''.join(c for c in v[2:] if c.isdigit() or c.lower() in 'abcdef')
			else:
				# For decimal values, only keep digits
				v_ret = ''.join(c for c in v if c.isdigit())
		return v_ret

	@staticmethod
	def _split_val_by_opertor(v):
		operators = ['(', ')', '+', '-', '*', '/', '%', '<<', '>>', '&', '|', '^']
		v_clean = v
		for op in operators: v_clean = v_clean.replace(op, ' ')
		v_clean = v_clean.split()
		return v_clean

	@staticmethod
	def _store_def(logger: logging.Logger, file_path: str, def_dump_df: pd.DataFrame):
		if def_dump_df.empty:
			logger.debug(f'No defines to store for file: {file_path}')
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
			f.writelines(def_dump_df['line'].to_list())
			file_closure = f'\n#endif  /* {file_define_guard_str} */\n'
			f.write(file_closure)
		logger.debug(f'Created file: {file_path}')

	def _create_main_include(self, file_path: str, soc_list: list):
		file_define_guard_str = f"{os.path.basename(file_path).upper().replace('.','_')}_"
		sub_series_include_selector_str = ''
		sorted_sub_series_list = [s.lower() for s in sorted(soc_list)]
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

	@staticmethod
	def _run_series_processing(logger: logging.Logger, compiler: str,
							input_files_list: list,
							dest_dir: str,
							series: str, soc: list,
							defines_list: list,
							include_path_list: list) -> list:
		start_time = datetime.now()
		series_dest_path = os.path.join(dest_dir, series, 'drivers', 'include')
		logger.debug(f'Use series destination folder: {series_dest_path}')
		os.makedirs(series_dest_path, exist_ok=True)

		logger.info(f'Processing SoC {soc}')
		logger.debug(f"Found headers (SoC: {soc}):\n\t{'\n\t'.join(input_files_list)}")
		full_soc_name = f'STM32{soc}'
		make_cmd_fn = lambda input_file: [compiler, '-dM', '-E', '-P', input_file] + \
											defines_list + include_path_list
		for input_file in input_files_list:
			cmd = make_cmd_fn(input_file)
			cmd_str = ' '.join(cmd)
			logger.debug(f'Execute command ({full_soc_name}): {cmd_str}')
			result = subprocess.run(cmd, capture_output=True, text=True, check=True)
			result_file_name_prefix = os.path.splitext(os.path.basename(input_file))[0]
			def_dump_df = gen_dts_defines._filter_defines_dump(logger, result.stdout)
			result_file_name = f'{result_file_name_prefix}_{full_soc_name.lower()}_def.dtsi'
			result_file = os.path.join(series_dest_path, result_file_name)
			gen_dts_defines._store_def(logger, result_file, def_dump_df)
			if len(result.stderr):
				logger.error(f'Command error (SoC: {full_soc_name}):\n{result.stderr}\nExecuted command: {cmd_str}\n')
				continue
			# TODO: Not all sub-series have all the peripherals, so the main include
			# file may include non-existing files. Need to filter out non-existing files.
			main_include_file = os.path.join(series_dest_path, f'{result_file_name_prefix}_def.dtsi')
			# self._create_main_include(main_include_file, soc_list)
		elapsed_time = (datetime.now() - start_time).total_seconds()
		logger.info(f'Completed SoC {soc} processing in {elapsed_time} sec.')
		return []

	def run(self):
		start_time = datetime.now()

		# Select series to process:
		if len(self._series) > 0:
			self._stm32_series_df = self._stm32_series_df[self._stm32_series_df['series'].isin(self._series)]

		self._lg.info(f"Processing SoC\'s:\n{self._stm32_series_df[['series', 'soc']]}")

		# Since this script extracts SoC peripherals defines from headers, there
		# is no interest in CPU headers used as a sub-dependency. Thus the CPU
		# headers may be leaved empty, but needed to be present in the include path.
		# Create empty core_XYZ.h headers:
		cpu_list = [
			'core_cm0', 'core_cm0plus', 'core_cm3', 'core_cm4', 'core_cm7',
			'core_cm33', 'core_cm55', 'core_cm85',
			'core_ca']
		self._core_headers_tmp_dir = tempfile.mkdtemp(
			prefix=f'stm32_includes_{self.__class__.__name__.lower()}_')
		self._lg.info(f'Created temp folder {self._core_headers_tmp_dir}')
		for cpu_name in cpu_list:
			header_path = os.path.join(self._core_headers_tmp_dir, cpu_name + '.h')
			with open(header_path, 'w', encoding='utf-8'): pass
			self._lg.debug(f'Created empty cpu-header: {header_path}')

		# Create a row with all defines used for each SoC:
		cpu_defines = [f'-D{cpu.upper()}' for cpu in cpu_list]
		self._stm32_series_df['defines'] = self._stm32_series_df.apply(
			lambda row: [f'-DSTM32{row["soc"]}'] + cpu_defines, axis=1)

		# Create row with includte patsh list for each SoC:
		self._stm32_series_df['include-path-list'] = self._stm32_series_df.apply(
			lambda row: [
				os.path.join(self._stm32cube_path, f'stm32{row["series"]}xx', 'drivers', 'include'),
				os.path.join(self._stm32cube_path, f'stm32{row["series"]}xx', 'soc'),
				self._core_headers_tmp_dir], axis=1)
		self._stm32_series_df['include-path-list'] = self._stm32_series_df['include-path-list'].apply(
			lambda x: sorted(set([f'-I{path}' for path in x])))

		# Define source directory for each series:
		self._stm32_series_df['src-dir'] = self._stm32_series_df.apply(
			lambda row: os.path.join(self._stm32cube_path,
							f"stm32{row['series']}xx", 'drivers','include'),
			axis=1)

		# Search _ll_ headers in each source-directory:
		self._stm32_series_df['input-files-list'] = self._stm32_series_df.apply(
			lambda row: sorted(set(glob.glob(f"{row['src-dir']}/*_ll_*.h"))), axis=1)

		# Create processing threads for each SoC series:
		self._stm32_series_df['thread'] = self._stm32_series_df.apply(
			lambda row: threading.Thread(target=self._run_series_processing,
				args=(self._lg, self._compiler, row['input-files-list'],
					self._dest_path, row['series'], row['soc'],
					row['defines'], row['include-path-list'])), axis=1)
		self._stm32_series_df.reset_index(drop=True, inplace=True)
		self._lg.debug(self._stm32_series_df)

		# Start all threads:
		self._stm32_series_df.apply(lambda row: row['thread'].start(), axis=1)

		# Wait for all threads to complete:
		self._stm32_series_df.apply(lambda row: row['thread'].join(), axis=1)

		elapsed_time = (datetime.now() - start_time).total_seconds()
		self._lg.info(f'Processing completed in {elapsed_time} sec.')


def main():
	# Define defalut arguments values:
	envZEPHYR_BASE = os.environ.get('ZEPHYR_BASE')
	stm32cube_path = os.path.abspath(
		os.path.join(envZEPHYR_BASE, '..', 'modules', 'hal', 'stm32', 'stm32cube'))
	compiler = shutil.which('clang')
	dest_path = os.path.abspath(os.path.join(stm32cube_path,'..','dts','st'))

	parser = argparse.ArgumentParser(description='Generate STM32 DTS definitions')
	parser.add_argument('-p', '--stm32cube-path', dest='stm32cube_path',
						default=stm32cube_path,
						help=f'Path to STM32Cube directory (default: \'{stm32cube_path}\')')
	parser.add_argument('-c', '--compiler', dest='compiler',
						default=compiler,
						help=f'Path to the compiler executable (default: \'{compiler}\')')
	parser.add_argument('-o', '--output-dir', dest='dest_path',
						default=dest_path,
						help=f'Output directory for generated files (default: \'{dest_path}\')')
	parser.add_argument('-s', '--series', dest='series',
						default=None, nargs='+',
						help='List of STM32 series to process (e.g., u3 h7). ' \
							 'If not specified, all series will be processed.')
	parser.add_argument('-v', '--verbose', action='store_true',
						help='Enable verbose output (makes execution slower)')
	args = parser.parse_args()

	log_level = logging.DEBUG if args.verbose else logging.INFO
	series = [] if args.series == None else [s.lower() for s in args.series]

	gdd = gen_dts_defines(args.stm32cube_path, args.dest_path, args.compiler,
						series, log_level=log_level)
	gdd.run()


if __name__ == '__main__':
	main()
