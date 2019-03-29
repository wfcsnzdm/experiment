
/*--------------------------------------------------------------------*/
/*--- Nulgrind: The minimal Valgrind tool.               nl_main.c ---*/
/*--------------------------------------------------------------------*/

/*
   This file is part of Nulgrind, the minimal Valgrind tool,
   which does no instrumentation or analysis.

   Copyright (C) 2002-2017 Nicholas Nethercote
      njn@valgrind.org

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
   02111-1307, USA.

   The GNU General Public License is contained in the file COPYING.
*/

#include "pub_tool_basics.h"
#include "pub_tool_aspacemgr.h"
#include "pub_tool_gdbserver.h"
#include "pub_tool_poolalloc.h"
#include "pub_tool_hashtable.h"     // For mc_include.h
#include "pub_tool_libcbase.h"
#include "pub_tool_libcassert.h"
#include "pub_tool_libcprint.h"
#include "pub_tool_machine.h"
#include "pub_tool_mallocfree.h"
#include "pub_tool_options.h"
#include "pub_tool_oset.h"
#include "pub_tool_rangemap.h"
#include "pub_tool_replacemalloc.h"
#include "pub_tool_tooliface.h"
#include "pub_tool_threadstate.h"
#include "pub_tool_xarray.h"
#include "pub_tool_xtree.h"
#include "pub_tool_xtmemory.h"
#include "ml_include.h"


static Bool none_handle_client_request ( ThreadId tid, UWord* arg, UWord* ret )
{
	switch (arg[0])
{
    	case VG_USERREQ__NONE_CREATE_SANDBOX:
        	VG_(printf)("1111111112311\n\n");
        	break;
	   case VG_USERREQ__NONE_TAINT:
		      VG_(printf)("111111212311111");
        	break;
    case VG_USERREQ__NONE_CREATE_NOEFFSET:
          VG_(printf)("33333333");
          break;
    case VG_USERREQ__NONE_CREATE_EFFSET:
          VG_(printf)("444444444444");
          break;
    case VG_USERREQ__MALTAINT_CALL_OPEN:
          VG_(printf)("555555");
          break;
    	default:
		VG_(printf)("none_handle_client_reques12t\n");
        	break;
}

  
  return True;
}

static void nl_post_clo_init(void)
{
}

static
IRSB* nl_instrument ( VgCallbackClosure* closure,
                      IRSB* bb,
                      const VexGuestLayout* layout, 
                      const VexGuestExtents* vge,
                      const VexArchInfo* archinfo_host,
                      IRType gWordTy, IRType hWordTy )
{
    return bb;
}

static void nl_fini(Int exitcode)
{
}

static void nl_pre_clo_init(void)
{
   VG_(printf)("test");
   VG_(details_name)            ("Nulgrind");
   VG_(details_version)         (NULL);
   VG_(details_description)     ("the minimal Valgrind tool");
   VG_(details_copyright_author)(
      "Copyright (C) 2002-2017, and GNU GPL'd, by Nicholas Nethercote.");
   VG_(details_bug_reports_to)  (VG_BUGS_TO);

   VG_(details_avg_translation_sizeB) ( 275 );
   VG_(needs_client_requests)(none_handle_client_request);
   VG_(basic_tool_funcs)        (nl_post_clo_init,
                                 nl_instrument,
                                 nl_fini);

   /* No needs, no core events to track */
}

VG_DETERMINE_INTERFACE_VERSION(nl_pre_clo_init)

/*--------------------------------------------------------------------*/
/*--- end                                                          ---*/
/*--------------------------------------------------------------------*/
