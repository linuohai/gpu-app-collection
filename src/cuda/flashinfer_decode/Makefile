ifeq ($(GPUAPPS_SETUP_ENVIRONMENT_WAS_RUN), 0)
$(error You must run "source setup_environment before calling make")
endif

OUTDIR := $(BINDIR)/$(BINSUBDIR)
PY_SCRIPT := $(OUTDIR)/flashinfer_decode.py
WRAPPERS := \
	$(OUTDIR)/flashinfer_decode_512 \
	$(OUTDIR)/flashinfer_decode_4096

.PHONY: all clean wrappers

all: $(PY_SCRIPT) wrappers

$(OUTDIR):
	mkdir -p $(OUTDIR)

$(PY_SCRIPT): flashinfer_decode.py | $(OUTDIR)
	cp $< $@
	chmod u+x $@

wrappers: $(WRAPPERS)
	chmod u+x $(WRAPPERS)

$(OUTDIR)/flashinfer_decode_512: flashinfer_decode_512 | $(OUTDIR)
	cp $< $@

$(OUTDIR)/flashinfer_decode_4096: flashinfer_decode_4096 | $(OUTDIR)
	cp $< $@

clean:
	@echo "clean: delete flashinfer_decode scripts from $(OUTDIR) manually if needed"
