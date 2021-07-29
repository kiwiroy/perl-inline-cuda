package Inline::CUDA::Utils;

use strict;
use warnings;

use Carp;
use File::Spec;
use Cwd;
use File::Temp qw/tempfile tempdir/;
use File::Which;
use File::ShareDir;
use Text::CSV_XS;

our $VERSION = '0.09';

our @MAKEFILE_PL_FLAGS_KEYS_COMPILER = qw/
	CCFLAGS
	CCCDLFLAGS
	CCDLFLAGS
	INC
	PERL_MALLOC_DEF
/;
our @MAKEFILE_PL_FLAGS_KEYS_LINKER = qw/
	LDDLFLAGS
	LDFLAGS
/;
our @MAKEFILE_PL_FLAGS_KEYS = (@MAKEFILE_PL_FLAGS_KEYS_COMPILER, @MAKEFILE_PL_FLAGS_KEYS_LINKER);

# parse the makefile for all 
sub enquire_Makefile_PL {
	my ($working_dir, $_mfile, $noisy) = @_;
	$noisy = 0 unless defined $noisy;

	my $parent = (caller(1))[3]; if( ! defined($parent) ){ $parent = 'N/A' }
	my $whoami = ( caller(0) )[3];

	my $old_dir = getcwd;

	if( ! -d $working_dir ){ print STDERR "$whoami (via $parent) : error, input working-dir '${working_dir}' is not valid dir.\n"; return undef }
	my $mfile = File::Spec->catdir($working_dir, $_mfile);
	if( ! -f $mfile ){ print STDERR "$whoami (via $parent) : error, '$mfile' is not a valid file: $!\n"; return undef }

	if( $noisy ){ print STDOUT "$whoami (via $parent) : file to read '$mfile' and current dir '$old_dir'.\n" }

	my $tmpdir = tempdir();
	if( ! -d $tmpdir ){ print STDERR "$whoami (via $parent) : error, call to tempdir() has failed.\n"; return undef }

	my $FH;
	if( ! open($FH, '<', $mfile) ){ print STDERR "$whoami (via $parent) : error, failed to open input file for reading '$mfile': $!\n"; return undef }
	my $contents; {local $/ = undef; $contents = <$FH> } close $FH;

	if( $noisy ){ print STDOUT "$contents\n\n$whoami (via $parent) : original '$mfile' contents above.\n" }

	# write the flags in this file, relative to current dir
	my $bingofile = File::Spec->catdir($tmpdir, 'bingofile.txt');

	# this assumes there is a line in the Makefile.PL like this:
	my $crucial_lineA = 'WriteMakefile(%options';
	my $crucial_lineB = ');';
	my $crucial_line = $crucial_lineA.$crucial_lineB;
	my $replacement_line = 'my $mm ='.${crucial_lineA}
		.", 'FIRST_MAKEFILE' => '__dummy123__'".${crucial_lineB}."\n"
		.'my $fh; if( ! open($fh, ">", "'.$bingofile.'") ){ die "error, failed to open file to write Makefile flags '."'".$bingofile."'".', $!" }'."\n"
		.'for my $k (sort qw/'.join(" ", @MAKEFILE_PL_FLAGS_KEYS).'/){'."\n"
		. <<'EOQ';
	my $v = $mm->{$k};
	chomp($v);
	#print STDOUT "bingobingo;;;;${k};;;;${v};;;;bingobingoend\n";
	print $fh "bingobingo;;;;${k};;;;${v};;;;bingobingoend\n";
}
close $fh;
EOQ
	$crucial_line = quotemeta($crucial_line);

	my $tmpMakefilename = File::Spec->catdir($tmpdir, 'Makefile');
	$replacement_line =~ s/__dummy123__/${tmpMakefilename}/;

	if( ! ($contents =~ s/^${crucial_line}/${replacement_line}/m) ){ print STDERR "$contents\n\n$whoami (via $parent) : error, did not find the crucial line '${crucial_line}' in Makefile.PL ('$mfile'), see contents above.\n"; return undef; }
	# write modified contents to a dummy Makefile.PL
	my $dummymfile = File::Spec->catdir($tmpdir, 'Makefile.PL');
	if( ! open($FH, '>', $dummymfile) ){ print STDERR "$whoami (via $parent) : error, failed to open existing Makefile.PL for writing back modified contents '$dummymfile': $!\n"; return undef }
	print $FH $contents; close $FH;

	if( $noisy ){ print STDOUT "$contents\n\n$whoami (via $parent) : new/modified '$dummymfile' contents above. Now changing to working dir '$working_dir' ...\n" }
	if( ! chdir $working_dir ){ print STDERR "$whoami (via $parent) : error, failed to change to the compilation working dir '$working_dir', $!\n"; return undef; }

	my $cmd = "$^X '${dummymfile}'";
	if( $noisy ){ print STDOUT "$whoami (via $parent) : at dir '$working_dir' executing command : ${cmd}\n" }
	#close($bingofh);
	my $output = '';
	my $stat = system($cmd);
#	my $output = qx/${cmd}/;
	if( $stat != 0 ){ print STDERR "$whoami (via $parent) : error, command has failed: ${cmd}\n"; return undef; }

	if( $noisy ){ print STDOUT "$output\n\n$whoami (via $parent) : success, output of making '$dummymfile' above with code $?.\n" }
	unlink $tmpMakefilename;

	my $fh; if( ! open($fh, '<', $bingofile) ){ print STDERR "$whoami (via $parent) : error, failed to open output file '$bingofile' produced by running command '$cmd' in dir '$working_dir', $!\n"; return undef }
	{ local $/ = undef; $output = <$fh> } close $fh;

	if( ! chdir $old_dir ){ print STDERR "$whoami (via $parent) : warning, failed to change to the initial working dir '$old_dir', $!\n"; return undef; }
	unlink $bingofile, $tmpMakefilename, $dummymfile;

	if( $noisy ){ print STDOUT "$output\n\n$whoami (via $parent) : success, contents of bingofile '$bingofile' containing Makefile flags, above.\n" }

	my %ret;
	for( split(/\n+/, $output) ){
		if( $_ =~ /^bingobingo;;;;(.+?);;;;(.+?);;;;bingobingoend$/ ){ $ret{$1} = $2 }
	}
	# and now split the string-of-flags (e.g. "-fPIC -Wall") into a hash keyed on each flag
	my $csv = Text::CSV_XS->new ({
		binary => 1, auto_diag => 1, sep_char => " ",
		quote_char  => undef, escape_char => undef
	});
	for my $k (keys %ret){
		my $K = 'hash-'.$k;
		my $flags_str = $ret{$k};
		if( ! $csv->parse($flags_str) ){ print STDERR "-----------\n$flags_str\n-------\n$whoami (via $parent) : error, call to csv->parse() has failed for key '$k' and above string, check the produce Makefile at '$tmpMakefilename' and '$dummymfile' (working dir: '$working_dir'): ".$csv->error_diag()."\n"; return undef }
		my @rows = $csv->fields();
		$ret{$K} = { map { $_ => 1 } grep(!/^\s*$/, @rows) };
	}
	if( $noisy ){
		print STDOUT "\n------------------------------\n$whoami (via $parent) : enquiry results of making '$dummymfile':\n";
		for ( grep(!/^hash-/, sort keys %ret) ){
			print STDOUT "  ".$_." => '".$ret{$_}."'\n";
		}
		for ( grep(/^hash-/, sort keys %ret) ){
			my $v = $ret{$_};
			for my $x (sort keys %$v){
				print STDOUT "  * <<<$x>>>\n";
			}
		}
		print STDOUT "$whoami (via $parent) : end enquiry results of making '$dummymfile'.\n------------------------------\n\n";
	}
	rmdir $tmpdir;
	return \%ret;
}

# find external executable dependencies (compilers)
# it first reads %ENV to see if user supplied a path during installation
# with something like CC=/usr/bin/gcc perl Makefile.PL
# if not, then it attempts to find the location of the executable based on some
# guessed names, e.g. cc, gcc. This works for the nvidia cuda compiler (nvcc) which
# has only one name but for compilers, it is a long list to be added here slowly.
# returns undef on failure or a hashref of the dependencies keyed on alias
# e.g. 'cc' => '/usr/bin/gcc'
# although the ENV keys are upper case (e.g. CC),
# the keys (not values!) to the returned hashref
# are all lowercase. e.g. 'cc' => '/xyz/Abc/123'
sub find_dependencies {
	my %ret;
	my ($aexe, $adir);

	my $parent = (caller(1))[3]; if( ! defined($parent) ){ $parent = 'N/A' }
	my $whoami = ( caller(0) )[3];

	$adir = envfind("CUDA_TOOLKIT_DIR");
	if( defined $adir ){
		for my $aexe ('nvcc', 'nvlink'){
			my $fullpath = File::Spec->catdir($adir, 'bin', $aexe);
			if( -x $fullpath ){ $ret{$aexe} = $fullpath }
		}
	}
	for my $aexe ('nvcc', 'nvlink'){
		if( ! exists $ret{$aexe} ){
			my $ucaexe = uc $aexe;
			my $aexe2 = envfind($ucaexe);
			if( ! defined $aexe2 ){
				# in m$-win it will search for nvcc.exe, nvcc.bat etc.
				$aexe2 = exefind($aexe);
				if( ! defined $aexe2 ){  print STDERR "$whoami (via $parent) : failed to find executable '$aexe' (NVIDIA CUDA Compiler) in path and '$ucaexe' environment variable was not set. If you do have this then either place it in a dir which is in PATH or specify the environment variable '$ucaexe' to point to that executable. For example (Linux,bash) $ucaexe=/usr/local/cuda/bin/$aexe perl Makefile.PL\n"; return undef }
			}
			if( ! -x $aexe2 ){ print STDERR "$whoami (via $parent) : failed, '$aexe' is not an executable.\n"; return undef }
			$ret{$aexe} = $aexe2;
		}
	}

	# we now have $ret{'nvcc'} and $ret{'nvlink'}, find their versions
	for my $aexe2 ('nvcc', 'nvlink'){
		$aexe = $ret{$aexe2};
		my $cmd = "${aexe} --version";
		my $retcom = qx/${cmd}/;
		if( $? ){ print STDERR (defined($retcom)?"$retcom\n\n":"")."$whoami (via $parent) : failed to find version of ${aexe} with this command: ${cmd}\n"; return undef }
		if( $retcom !~ /^Cuda\s+compilation\s+tools,\s+release\s+([^\s]+),\s+([^\s]+)\s*$/m ){ print STDERR "$retcom\n\n$whoami (via $parent) : error, failed to parse version string in above output of this command: ${cmd}\n"; return undef }
		$ret{$aexe2.'-release'} = $1;
		$ret{$aexe2.'-version'} = $2;
	}

	$aexe = envfind("CC");
	if( ! defined $aexe ){
		$aexe = exefind('cc') or exefind('gcc');
		if( ! defined $aexe ){ print STDERR "$whoami (via $parent) : failed to find executable 'cc' or 'gcc' (C Compiler) in path and CC environment variable was not set. If you do have this then either place it in a dir which is in PATH or specify the environment variable CC to point to that executable. For example (Linux,bash) CC=/usr/bin/cc perl Makefile.PL\n"; return undef }
	}
	if( ! -x $aexe ){ print STDERR "$whoami (via $parent) : failed, '$aexe' is not an executable.\n"; return undef }
	$ret{cc} = $aexe;

	$aexe = envfind("LD");
	if( ! defined $aexe ){
		$aexe = exefind('cc') or exefind('gcc');
		if( ! defined $aexe ){ print STDERR "$whoami (via $parent) : failed to find executable 'cc' or 'gcc' (C Linker) in path and LD environment variable was not set. If you do have this then either place it in a dir which is in PATH or specify the environment variable LD to point to that executable. For example (Linux,bash) LD=/usr/bin/cc perl Makefile.PL\n"; return undef }
	}
	if( ! -x $aexe ){ print STDERR "$whoami (via $parent) : failed, '$aexe' is not an executable.\n"; return undef }
	$ret{ld} = $aexe;

	$aexe = envfind("CXX");
	if( ! defined $aexe ){
		$aexe = exefind('c++') or exefind('g++');
		if( ! defined $aexe ){ print STDERR "$whoami (via $parent) : failed to find executable 'c++' or 'g++' (C++ Compiler) in path and CXX environment variable was not set. If you do have this then either place it in a dir which is in PATH or specify the environment variable CXX to point to that executable. For example (Linux,bash) CXX=/usr/bin/g++ perl Makefile.PL\n"; return undef }
	}
	if( ! -x $aexe ){ print STDERR "$whoami (via $parent) : failed, '$aexe' is not an executable.\n"; return undef }
	$ret{cxx} = $aexe;

	return \%ret
}

sub envfind { return exists($ENV{$_[0]}) && defined($ENV{$_[0]}) ? $ENV{$_[0]} : undef }
sub exefind { return File::Which::which($_[0]) }

# save the dependencies hash (e.g. cc=>'/usr/bin/cc' etc.) into the specified file ($conf_file)
# which is optional. If no file is specified, then the path_of_configuration_file_in_shared_dir()
# will be called to provide us with the perl's system shared dir for our current app
# returns 1 on failure or 0 on success
sub save_configuration {
	my ($conf, $conf_file) = @_;

	my $parent = (caller(1))[3]; if( ! defined($parent) ){ $parent = 'N/A' }
	my $whoami = ( caller(0) )[3];

	if( ! defined $conf_file ){
		if( ! defined($conf_file=path_of_configuration_file_in_shared_dir()) ){ print STDERR "$whoami (via $parent) : error, call to path_of_configuration_file_in_shared_dir() has failed.\n"; return 1 }
	}

	my $fh;
	if( ! open($fh, '>', $conf_file) ){ print STDERR "$whoami (via $parent) : error, failed to open file '$conf_file' for writing, $!\n"; return 1 }
	for my $k (sort keys %$conf){
		if( ($k =~ /[=]/) || ($conf->{$k} =~ /[=]/) ){ print STDERR "$whoami (via $parent) : error, configuration line '$k' => '$conf->{$k}' contains illegal characters.\n"; return 1; }
		print $fh $k."=".$conf->{$k}."\n";
	}
	close($fh);
	return 0;
}
# load configuration from specified configuration file
# or the default shared-dir/<conf> (perl knows) if no $conf_file is specified
# see path_of_configuration_file_in_shared_dir() which returns the full-path
# to (a candidate) <conf> file.
# returns the configuration as a dependencies hash, e.g. 'cc' => '/usr/bin/cc' etc. on success
# returns undef on failure
sub load_configuration {
	my $conf_file = shift;

	my $parent = (caller(1))[3]; if( ! defined($parent) ){ $parent = 'N/A' }
	my $whoami = ( caller(0) )[3];

	if( ! defined $conf_file ){
		if( ! defined($conf_file=path_of_configuration_file_in_shared_dir()) ){ print STDERR "$whoami (via $parent) : error, call to path_of_configuration_file_in_shared_dir() has failed.\n"; return 1 }
	}

	my $fh;
	my %ret;
	if( ! open($fh, '<', $conf_file) ){ print STDERR "$whoami (via $parent) :  error, failed to open file '$conf_file' for reading, $!\n"; return undef }
	my $LN = 0;
	while( my $aline = <$fh> ){
		$LN++;
		chomp($aline);
		$aline =~ s/#.*$//;
		if( $aline =~ /^\s*$/ ){ next }
		my @items = split /\s*=\s*/, $aline;
		if( scalar(@items) == 0 ){ print STDERR "$whoami (via $parent) : error, file '$conf_file', line $LN : no '='.\n"; return undef }
		if( scalar(@items) > 2 ){ print STDERR "$whoami (via $parent) : error, file '$conf_file', line $LN : '=' is illegal character.\n"; return undef }
		$ret{$items[0]} = defined($items[1]) ? $items[1] : '';
	}
	close($fh);
	return \%ret;
}
# return the path to the configuration file installed during installation
# is some shared dir that perl knows, in linux this is like <sys>/auto/share/dist/Inline-CUDA/Inline-CUDA.conf
# it uses File::ShareDir::dist_dir() to find the root path
# return undef on failure
sub path_of_configuration_file_in_shared_dir {
	my $dist_name = $_[0];

	my $parent = (caller(1))[3]; if( ! defined($parent) ){ $parent = 'N/A' }
	my $whoami = ( caller(0) )[3];

	if( ! defined $dist_name ){
		$dist_name = __PACKAGE__;
		if( ! ($dist_name =~ s/::Utils$//) ){ print STDERR "$whoami (via $parent) : You changed the package name but you forgot to update me!"; return undef }
		$dist_name =~ s/::/-/g; # silly
	}
	my $conf_file_base = $dist_name . '.conf';
	my $dist_dir = File::ShareDir::dist_dir($dist_name);
	if( ! defined $dist_dir ){ print STDERR "$whoami (via $parent) : failed to find dist_dir (where the configuration file must reside), call to File::ShareDir::dist_dir() has failed."; return undef }
	my $conf_file = File::Spec->catdir($dist_dir, $conf_file_base);
	# last paranoid test
	if( ! -e $conf_file ){ print STDERR "$whoami (via $parent) : configuration file '$conf_file' does not exist on disk."; return undef }
	return $conf_file;
}
# find and read the configuration file which was installed during installation
# and parse and return its contents as a hashref
# return undef on failure
sub read_configuration_file_in_shared_dir {
	my $conf_file = path_of_configuration_file_in_shared_dir();

	my $parent = (caller(1))[3]; if( ! defined($parent) ){ $parent = 'N/A' }
	my $whoami = ( caller(0) )[3];

	if( ! defined $conf_file ){ print STDERR "$whoami (via $parent) : call to path_of_configuration_file_in_shared_dir() has failed."; return undef }
	my $conf = load_configuration($conf_file);
	if( ! defined $conf ){ print STDERR "$whoami (via $parent) : call to load_configuration() has failed for file '$conf_file'."; return undef }
	return $conf
}

# attempts to do a basic compilation just to see if nvidia-cuda-compiler (nvcc)
# approves the found C compiler
# the same lame test is also done during "make test" and exists under t/01-config.t
# the input parameter ($deps) is a hash-ref of configuration as read from the configuration
# file (see load_configuration())
sub test_compilers {
	#a hashref of conf as returned by Inline::CUDA::Utils::load_configuration()
	my $conf = $_[0];

	my $parent = (caller(1))[3]; if( ! defined($parent) ){ $parent = 'N/A' }
	my $whoami = ( caller(0) )[3];

	if( ! defined $conf ){
		if( ! defined($conf=load_configuration()) ){ print STDERR "$whoami (via $parent) : error, call to load_configuration() has failed.\n"; return 1 }
	}

	my ($cc, $cxx, $ld, $nvcc);
	if( ! exists($conf->{'cc'}) || ! defined($cc=$conf->{'cc'}) ){ print STDERR "$whoami (via $parent) : error, no C compiler has been specified, no key 'cc' in configuration hash."; return 1 }
	if( ! exists($conf->{'cxx'}) || ! defined($cxx=$conf->{'cxx'}) ){ print STDERR "$whoami (via $parent) : error, no C compiler has been specified, no key 'cxx' in configuration hash."; return 1 }
	if( ! exists($conf->{'ld'}) || ! defined($ld=$conf->{'ld'}) ){ print STDERR "$whoami (via $parent) : error, no C compiler has been specified, no key 'ld' in configuration hash."; return 1 }
	if( ! exists($conf->{'nvcc'}) || ! defined($nvcc=$conf->{'nvcc'}) ){ print STDERR "$whoami (via $parent) : error, no C compiler has been specified, no key 'nvcc' in configuration hash."; return 1 }

	my ($tmpfh, $tmpfilename) = tempfile(SUFFIX => '.cu');
	if( ! defined $tmpfh ){ print STDERR "$whoami (via $parent) : error, call to tempfile() has failed!"; return 1 }
	print $tmpfh <<'EOCUDA';
#include <stdio.h>
#define N 10
__global__
void lametest(int *a) {
    int i = blockIdx.x;
    if (i<N) a[i] = 2.0 * a[i];
}
int main(){
	cudaError_t err;
	int ha[N], i, *da;
	for(i=0;i<N;i++) ha[i] = i;
	if( (err=cudaMalloc((void **)&da, N*sizeof(int))) != cudaSuccess ){ fprintf(stderr, "main(): error, call to cudaMalloc() has failed for %zu bytes for db: %s\n", N*sizeof(int), cudaGetErrorString(err)); return 1; }
	if( (err=cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess ){ fprintf(stderr, "main(): error, call to cudaMemcpy(cudaMemcpyHostToDevice) has failed for %zu bytes for ha->da: %s\n", N*sizeof(int), cudaGetErrorString(err)); return 1; }
	// triple angle brackets !!! <<< >>>
	lametest<<<N,1>>>(da);
	if( (err=cudaGetLastError()) != cudaSuccess ){ fprintf(stderr, "main(): error, failed to launch the kernel into the device: %s\n", cudaGetErrorString(err)); return 1; }
	return 0;
}
EOCUDA
	my $cmd = "$nvcc --compiler-bindir '$cxx' '$tmpfilename'";
	my $retcom = qx/$cmd/;
	my $status = $?;
	if( $status != 0 ){ print STDERR "\n$retcom\n\n$whoami (via $parent) : error, command has failed with code '$status' and above output: $cmd\n"; close($tmpfh); unlink($tmpfilename); return 1 }
	print STDERR "$retcom\n\ntest_compilers() : done compiled a small cuda/c program with above output.";
	close($tmpfh); unlink($tmpfilename);
	return 0;
}
1;
