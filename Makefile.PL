use 5.006;
use strict;
use warnings;
use ExtUtils::MakeMaker;
use File::ShareDir::Install;
use File::Spec;
use FindBin;

my %WriteMakefileArgs = (
	NAME		   => 'Inline::CUDA',
	AUTHOR		   => q{Andreas Hadjiprocopis <bliako@cpan.org>},
	VERSION_FROM	   => 'lib/Inline/CUDA.pm',
	ABSTRACT_FROM	   => 'lib/Inline/CUDA.pm',
	EXE_FILES 	   => [qw!scripts/nvidia-ml-test.pl!],
	LICENSE		   => 'artistic_2',
	MIN_PERL_VERSION   => '5.006',
	CONFIGURE_REQUIRES => {
		'ExtUtils::MakeMaker' => '0',
		'File::Which' => '0',
		'File::Find' => '0',
		'FindBin' => '0',
		'File::Spec' => '0',
		'File::ShareDir::Install' => '0',
	},
	META_MERGE => {
		'meta-spec' => { version => 2 },
		requires	=> { perl	=> '5.10.0' },
		resources   => {
			license	=> 'artistic_2',
			homepage   => 'https://metacpan.org/module/Inline::CUDA',
			repository => {
				url  => 'https://github.com/hadjiprocopis/perl-inline-cuda',
				web  => 'https://github.com/hadjiprocopis/perl-inline-cuda',
				type => 'git',
			},
			bugtracker => {
				web => 'https://github.com/hadjiprocopis/perl-inline-cuda/issues',
			}
		},
	},
	TEST_REQUIRES => {
		'Test::More' => '0',
	},
	PREREQ_PM => {
		#'Foo::Bar::Module' => '5.0401',
		'Inline' => '0.56',
		'Inline::C' => '0.80',
		'Data::Dumper' => '0',
		'Config' => '0',
		'Carp' => '0',
		'Cwd' => '0',
		'File::Spec' => '0',
		'File::Copy' => '0',
		'File::Temp' => '0',
		'File::ShareDir' => '0',
		'Test::More' => '0',
		'Test::Pod' => '0',
		'Test::CheckManifest' => '0',
		'Time::HiRes' => '0',
		'Data::Dumper' => '0',
		'Math::Matrix' => '0',
	},
	dist  => { COMPRESS => 'gzip -9f', SUFFIX => 'gz', },
	clean => { FILES => 'Inline-CUDA-* _Inline' },
	postamble => { BENCHMARK_FILES => 'xt/benchmarks/*.b' }
);

require File::Spec->catdir($FindBin::Bin, 'lib', 'Inline', 'CUDA', 'Utils.pm');

# install all files in config dir (they will be created below) (call before WriteMakefile())
File::ShareDir::Install::install_share dist => File::Spec->catdir($FindBin::Bin, 'config');

my $dependencies = Inline::CUDA::Utils::find_dependencies();
if( ! defined $dependencies ){ print STDERR "$0 : error, call to Inline::CUDA::Utils::find_dependencies() has failed.\n"; exit(1) }
print "find_dependencies() : report:\n"; print "  $_ : ".$dependencies->{$_}."\n" for (sort keys %$dependencies);

my $dist_name = $WriteMakefileArgs{NAME}; $dist_name =~ s/::/-/g;
my $conf_file_base = $dist_name . '.conf';
my $conf_file = File::Spec->catdir($FindBin::Bin, 'config', $conf_file_base);
if( Inline::CUDA::Utils::save_configuration($dependencies, $conf_file) ){ print STDERR "$0 : error, call to Inline::CUDA::Utils::save_configuration() has failed to save into '$conf_file'.\n"; exit(1) }
print "$0 : configuration written to '$conf_file'.\n";

my $newconf = Inline::CUDA::Utils::load_configuration($conf_file);
if( ! defined $newconf ){ print STDERR "$0 : error, call to Inline::CUDA::Utils::load_configuration() has failed for file '$conf_file'. Can not verify that the configuration file was written correctly.\n"; exit(1) }

# Compatibility with old versions of ExtUtils::MakeMaker
unless (eval { ExtUtils::MakeMaker->VERSION('6.64'); 1 }) {
	my $test_requires = delete $WriteMakefileArgs{TEST_REQUIRES} || {};
	@{$WriteMakefileArgs{PREREQ_PM}}{keys %$test_requires} = values %$test_requires;
}

unless (eval { ExtUtils::MakeMaker->VERSION('6.55_03'); 1 }) {
	my $build_requires = delete $WriteMakefileArgs{BUILD_REQUIRES} || {};
	@{$WriteMakefileArgs{PREREQ_PM}}{keys %$build_requires} = values %$build_requires;
}

delete $WriteMakefileArgs{CONFIGURE_REQUIRES}
	unless eval { ExtUtils::MakeMaker->VERSION('6.52'); 1 };
delete $WriteMakefileArgs{MIN_PERL_VERSION}
	unless eval { ExtUtils::MakeMaker->VERSION('6.48'); 1 };
delete $WriteMakefileArgs{LICENSE}
	unless eval { ExtUtils::MakeMaker->VERSION('6.31'); 1 };

WriteMakefile(%WriteMakefileArgs);

{
package MY;
use File::ShareDir::Install;
 
sub libscan {
    my( $mm, $file ) = @_;
    return if $file =~ /(push_to_GIT)|(\.lock)/;
    warn "@_\n";
    return $file;
}

sub postamble {
	my ($self, %h) = @_;
	my @ret = File::ShareDir::Install::postamble( $self );
	# ... add more things to @ret;
	my $retstr = join "\n", @ret;
	# WARNING: USE REAL TABS in the Here-doc below, it goes into Makefile!!!
	$retstr .= "BENCHMARK_FILES=$h{BENCHMARK_FILES}\n"
	. <<'POSTAMBLE';
TEST_D = $(ABSPERLRUN) -MExtUtils::Command -e test_d --

benchmark :: $(BENCHMARK_FILES)
	prove --blib $(INST_LIB) --blib $(INST_ARCHLIB) --verbose $^

bench2 :: $(BENCHMARK_FILES)
	$(TEST_D) xt && $(MAKE) test TEST_FILES='$(BENCHMARK_FILES)'

author-test :: all xt/*.t
	prove -Iblib/lib -vl xt/*.t

POSTAMBLE


}
}
1;
