"""
Єдиний інтерфейс для лабораторних 1-6.
"""
import base64
import os
import tempfile
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from core.color import for_display, gray_to_bgr

try:
    from streamlit_iframe_event import st_iframe_event
except Exception:
    st_iframe_event = None

st.set_page_config(
    page_title="Лабораторні з CV",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Папка з тестовими зображеннями
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test_images")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

LAB_MODULES = {
    1: ("labs.lab1", "Лаб 1: Вступ до OpenCV"),
    2: ("labs.lab2", "Лаб 2: Колірні моделі та пікселі"),
    3: ("labs.lab3", "Лаб 3: Фільтрація, краї, морфологія"),
    4: ("labs.lab4", "Лаб 4: Сегментація, ключові точки"),
    5: ("labs.lab5", "Лаб 5: Відео та вебкамера"),
    6: ("labs.lab6", "Лаб 6: Кольорова сегментація"),
}


def _lab3_kwargs_from_session(task_idx: int) -> dict:
    """Зібрати kwargs для лаб 3 з session_state."""
    out = {}
    if task_idx in (0, 1, 2, 3):
        out["ksize"] = st.session_state.get("lab3_ksize", 5)
    if task_idx == 1:
        out["sigma_x"] = st.session_state.get("lab3_sigma", 0.0)
    if task_idx == 5:
        out["ksize"] = int(st.session_state.get("lab3_edge_ksize", 3))
    if task_idx in (6, 7, 8, 9):
        out["low"] = st.session_state.get("lab3_canny_low", 50)
        out["high"] = st.session_state.get("lab3_canny_high", 150)
    if task_idx == 10:
        out["thresh"] = st.session_state.get("lab3_thresh", 127)
    if task_idx in (11, 12, 13, 14):
        out["thresh"] = st.session_state.get("lab3_morph_thresh", 127)
    if task_idx in (11, 12, 13):
        out["ksize"] = st.session_state.get("lab3_morph_ksize", 3)
    return out


def _lab4_kwargs_from_session(task_idx: int) -> dict:
    """Зібрати kwargs для лаб 4 з session_state."""
    out = {}
    if task_idx == 0:
        out["thresh"] = st.session_state.get("lab4_thresh", 127)
    if task_idx == 1:
        out["block_size"] = st.session_state.get("lab4_block_size", 11)
        out["C"] = st.session_state.get("lab4_C", 2)
    if task_idx == 3:
        out["K"] = st.session_state.get("lab4_K", 3)
    if task_idx == 5:
        out["block_size"] = st.session_state.get("lab4_harris_block", 2)
        out["ksize"] = st.session_state.get("lab4_harris_ksize", 3)
        out["k"] = st.session_state.get("lab4_harris_k", 0.04)
    return out


# Стилі для кращої читабельності та ієрархії
st.markdown("""
<style>
    /* Менше відступу після заголовків у сайдбарі */
    [data-testid="stSidebar"] .stMarkdown { margin-bottom: 0.25rem; }
    /* Відступ між секціями результатів */
    .result-block { margin-bottom: 2rem; }
    /* Підпис під зображенням */
    .image-caption { font-size: 0.9rem; color: #666; margin-top: 0.25rem; }
    /* Акцент для головної кнопки в сайдбарі */
    [data-testid="stSidebar"] button[kind="primary"] { font-weight: 600; }
    /* Компактніший file uploader */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] { padding: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)


def get_test_image_files():
    if not os.path.isdir(TEST_IMAGES_DIR):
        return []
    files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(IMAGE_EXTENSIONS)]
    return sorted(files)


def get_lab_module(lab_num):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        f"lab{lab_num}",
        os.path.join(os.path.dirname(__file__), "labs", f"lab{lab_num}.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_task_with_image(lab_num, task_idx, image_path, **kwargs):
    mod = get_lab_module(lab_num)
    tasks = mod.get_tasks()
    _, fn = tasks[task_idx]
    if lab_num == 5:
        return fn(**kwargs)
    return fn(image_path, **kwargs)


def _pixel_tracker_html(w, h, channels, b64_esc):
    """Генерує HTML+JS для відображення зображення та RGB/RGBA під курсором у реальному часі (як у головному полі)."""
    return f"""
<div id="pixel-tracker-root">
  <canvas id="pt-canvas" width="{w}" height="{h}" style="max-width:100%; height:auto; display:block;"></canvas>
  <p id="pt-value" style="margin-top:0.5rem; font-family:monospace; font-size:0.95rem; color:#333;">Наведіть курсор на зображення</p>
</div>
<script>
(function() {{
  var W = {w}, H = {h}, CH = {channels};
  var b64 = "{b64_esc}";
  var bin = atob(b64);
  var buf = new Uint8Array(bin.length);
  for (var i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);

  var canvas = document.getElementById('pt-canvas');
  var ctx = canvas.getContext('2d');
  var imgData = ctx.createImageData(W, H);

  if (CH === 1) {{
    for (var i = 0; i < W * H; i++) {{
      var v = buf[i];
      imgData.data[i*4] = imgData.data[i*4+1] = imgData.data[i*4+2] = v;
      imgData.data[i*4+3] = 255;
    }}
  }} else if (CH === 4) {{
    for (var i = 0; i < W * H; i++) {{
      var j = i * 4;
      imgData.data[i*4] = buf[j+2];
      imgData.data[i*4+1] = buf[j+1];
      imgData.data[i*4+2] = buf[j];
      imgData.data[i*4+3] = buf[j+3];
    }}
  }} else {{
    for (var i = 0; i < W * H; i++) {{
      var j = i * 3;
      imgData.data[i*4] = buf[j+2];
      imgData.data[i*4+1] = buf[j+1];
      imgData.data[i*4+2] = buf[j];
      imgData.data[i*4+3] = 255;
    }}
  }}
  ctx.putImageData(imgData, 0, 0);

  var label = document.getElementById('pt-value');
  function showPixel(ev) {{
    var rect = canvas.getBoundingClientRect();
    var x = Math.floor((ev.clientX - rect.left) / rect.width * W);
    var y = Math.floor((ev.clientY - rect.top) / rect.height * H);
    if (x < 0 || x >= W || y < 0 || y >= H) {{ label.textContent = 'Поза зображенням'; return; }}
    if (CH === 1) {{
      label.textContent = '(' + x + ', ' + y + ') значення = ' + buf[y * W + x];
    }} else if (CH === 4) {{
      var j = (y * W + x) * 4;
      label.textContent = '(' + x + ', ' + y + ') RGBA = (' + buf[j+2] + ', ' + buf[j+1] + ', ' + buf[j] + ', ' + buf[j+3] + ')';
    }} else {{
      var j = (y * W + x) * 3;
      label.textContent = '(' + x + ', ' + y + ') RGB = (' + buf[j+2] + ', ' + buf[j+1] + ', ' + buf[j] + ')';
    }}
  }}
  canvas.addEventListener('mousemove', showPixel);
  canvas.addEventListener('mouseleave', function() {{ label.textContent = 'Наведіть курсор на зображення'; }});
}})();
</script>
"""


def _decode_data_url_to_bytes(data_url):
    """Повертає bytes з data URL (data:image/png;base64,...) або None."""
    if not data_url or not isinstance(data_url, str):
        return None
    if not data_url.startswith("data:"):
        return None
    try:
        base64_str = data_url.split(",", 1)[1]
        return base64.b64decode(base64_str)
    except Exception:
        return None


def _draw_rectangles_html_sync(w, h, r, g, b, channels=None, b64_esc=None, initial_data_url=None):
    """HTML для iframe: малювання прямокутників; при mouseup надсилає canvas у parent через postMessage.
    Якщо initial_data_url задано — спочатку завантажуємо зображення з неї; інакше рисуємо з raw b64.
    Кнопки «Зберегти» тут немає — вона в сайдбарі.
    """
    if initial_data_url:
        # Екрануємо для JS-рядка (backslash, лапки, теги)
        url_esc = (
            initial_data_url.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("</script>", "<\\/script>")
        )
        init_block = f"""
  var img = new Image();
  img.onload = function() {{
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);
    imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    var W = canvas.width, H = canvas.height;
    attachDraw();
    if (window.parent && window.parent !== window) {{ try {{ window.parent.postMessage(canvas.toDataURL('image/png'), '*'); }} catch(e) {{}} }}
  }};
  img.src = "{url_esc}";
"""
        raw_block = ""
    else:
        init_block = ""
        raw_block = f"""
  var W = {w}, H = {h}, CH = {channels};
  var b64 = "{b64_esc}";
  var bin = atob(b64);
  var buf = new Uint8Array(bin.length);
  for (var i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  var imgData = ctx.createImageData(W, H);
  if (CH === 1) {{
    for (var i = 0; i < W * H; i++) {{
      var v = buf[i];
      imgData.data[i*4] = imgData.data[i*4+1] = imgData.data[i*4+2] = v;
      imgData.data[i*4+3] = 255;
    }}
  }} else if (CH === 4) {{
    for (var i = 0; i < W * H; i++) {{
      var j = i * 4;
      imgData.data[i*4] = buf[j];
      imgData.data[i*4+1] = buf[j+1];
      imgData.data[i*4+2] = buf[j+2];
      imgData.data[i*4+3] = buf[j+3];
    }}
  }} else {{
    for (var i = 0; i < W * H; i++) {{
      var j = i * 3;
      imgData.data[i*4] = buf[j];
      imgData.data[i*4+1] = buf[j+1];
      imgData.data[i*4+2] = buf[j+2];
      imgData.data[i*4+3] = 255;
    }}
  }}
  ctx.putImageData(imgData, 0, 0);
  attachDraw();
  if (window.parent && window.parent !== window) {{ try {{ window.parent.postMessage(canvas.toDataURL('image/png'), '*'); }} catch(e) {{}} }}
"""

    return f"""
<!DOCTYPE html>
<html>
<body style="margin:0;">
<div id="draw-rect-root">
  <canvas id="dr-canvas" width="{w}" height="{h}" style="max-width:100%; height:auto; display:block; cursor:crosshair;"></canvas>
  <p id="dr-hint" style="margin-top:0.25rem; font-size:0.9rem; color:#666;">Потягніть мишкою, щоб намалювати прямокутник. Зберегти — у сайдбарі.</p>
</div>
<script>
(function() {{
  var R = {r}, G = {g}, B = {b};
  var canvas = document.getElementById('dr-canvas');
  var ctx = canvas.getContext('2d');
  var imgData = null;
  var W = {w}, H = {h};

  function attachDraw() {{
    if (!imgData) return;
    W = canvas.width;
    H = canvas.height;

    function fillRect(x1, y1, x2, y2) {{
      for (var y = y1; y < y2; y++) {{
        for (var x = x1; x < x2; x++) {{
          var i = (y * W + x) * 4;
          imgData.data[i] = R;
          imgData.data[i+1] = G;
          imgData.data[i+2] = B;
          imgData.data[i+3] = 255;
        }}
      }}
      ctx.putImageData(imgData, 0, 0);
      if (window.parent && window.parent !== window) {{
        try {{ window.parent.postMessage(canvas.toDataURL('image/png'), '*'); }} catch(e) {{}}
      }}
    }}
    var startX = null, startY = null;
    function toImgCoords(ev) {{
      var rect = canvas.getBoundingClientRect();
      var x = Math.floor((ev.clientX - rect.left) / rect.width * W);
      var y = Math.floor((ev.clientY - rect.top) / rect.height * H);
      return {{ x: Math.max(0, Math.min(W-1, x)), y: Math.max(0, Math.min(H-1, y)) }};
    }}
    canvas.addEventListener('mousedown', function(ev) {{
      var p = toImgCoords(ev);
      startX = p.x; startY = p.y;
    }});
    canvas.addEventListener('mousemove', function(ev) {{
      if (startX === null) return;
      var p = toImgCoords(ev);
      ctx.putImageData(imgData, 0, 0);
      ctx.strokeStyle = 'rgba(255,255,0,0.9)';
      ctx.lineWidth = 2;
      ctx.strokeRect(Math.min(startX,p.x), Math.min(startY,p.y), Math.abs(p.x-startX)+1, Math.abs(p.y-startY)+1);
    }});
    canvas.addEventListener('mouseup', function(ev) {{
      if (startX === null) return;
      var p = toImgCoords(ev);
      var x1 = Math.min(startX, p.x), x2 = Math.max(startX, p.x)+1;
      var y1 = Math.min(startY, p.y), y2 = Math.max(startY, p.y)+1;
      startX = null; startY = null;
      if (x2 <= x1 || y2 <= y1) return;
      fillRect(x1, y1, x2, y2);
    }});
    canvas.addEventListener('mouseleave', function() {{ startX = null; startY = null; if (imgData) ctx.putImageData(imgData, 0, 0); }});
  }}
  {init_block}
  {raw_block}
}})();
</script>
</body>
</html>
"""


def _draw_rectangles_html(w, h, channels, b64_esc, r, g, b, reset_key=0):
    """Canvas + вибір кольору, Скинути та Зберегти; стилі беруться з теми Streamlit (батьківське вікно)."""
    # Резервні стилі кнопок, щоб завжди виглядали як кнопки (не текст); потім JS підставить тему
    btn_default = (
        "padding:0.25rem 0.75rem; font-size:0.875rem; font-family:inherit; cursor:pointer; "
        "border-radius:0.5rem; border:1px solid rgba(128,128,128,0.4); "
        "background-color:#f0f2f6; color:rgb(49,51,63); "
        "appearance:button; -webkit-appearance:button; box-shadow:0 1px 2px rgba(0,0,0,0.05);"
    )
    return f"""
<style>body{{margin:0;padding:0;}} #draw-rect-root{{width:100%;}} #dr-toolbar{{width:100%; box-sizing:border-box;}}</style>
<div id="draw-rect-root" data-reset="{reset_key}" style="width:100%; box-sizing:border-box;">
  <canvas id="dr-canvas" width="{w}" height="{h}" style="max-width:100%; height:auto; display:block; cursor:crosshair;"></canvas>
  <div id="dr-toolbar" style="width:100%; box-sizing:border-box; margin:0; padding:0.75rem 1rem; display:flex; align-items:center; justify-content:center; gap:0.75rem; flex-wrap:wrap;">
    <label id="dr-color-label" style="display:flex; align-items:center; gap:0.35rem; font-size:0.875rem;">
      Колір: <input type="color" id="dr-color" value="#{r:02x}{g:02x}{b:02x}" style="width:2rem; height:1.5rem; cursor:pointer; border-radius:0.5rem;">
    </label>
    <button type="button" id="dr-reset" style="{btn_default}">Скинути малюнок</button>
    <button type="button" id="dr-save" style="{btn_default}">Зберегти зображення</button>
  </div>
</div>
<script>
(function() {{
  function applyStreamlitStyles() {{
    try {{
      var parent = window.parent;
      if (!parent || !parent.document) return;
      var root = parent.document.documentElement;
      var sidebar = parent.document.querySelector('[data-testid="stSidebar"]');
      var btn = parent.document.querySelector('[data-testid="stSidebar"] button') || parent.document.querySelector('.stButton > button') || parent.document.querySelector('button');
      var style = root ? parent.getComputedStyle(root) : null;
      var sidebarStyle = sidebar ? parent.getComputedStyle(sidebar) : null;
      var btnStyle = btn ? parent.getComputedStyle(btn) : null;
      var toolbarBg = (sidebarStyle && sidebarStyle.backgroundColor) ? sidebarStyle.backgroundColor : (style ? (style.getPropertyValue('--secondary-background-color') || style.backgroundColor) : '#f0f2f6');
      var toolbar = document.getElementById('dr-toolbar');
      if (toolbar) toolbar.style.backgroundColor = toolbarBg;
      var bg = (btnStyle && btnStyle.backgroundColor) ? btnStyle.backgroundColor : (style ? (style.getPropertyValue('--secondary-background-color') || style.backgroundColor) : '');
      var color = (btnStyle && btnStyle.color) ? btnStyle.color : (style ? (style.getPropertyValue('--text-color') || style.color) : '');
      var font = (btnStyle && btnStyle.fontFamily) ? btnStyle.fontFamily : (style ? style.fontFamily : 'inherit');
      var radius = (btnStyle && btnStyle.borderRadius) ? btnStyle.borderRadius : '0.5rem';
      var padding = (btnStyle && btnStyle.padding) ? btnStyle.padding : '0.25rem 0.75rem';
      var border = (btnStyle && btnStyle.border) ? btnStyle.border : '1px solid rgba(128,128,128,0.3)';
      var boxShadow = (btnStyle && btnStyle.boxShadow) ? btnStyle.boxShadow : 'none';
      var fontSize = (btnStyle && btnStyle.fontSize) ? btnStyle.fontSize : '0.875rem';
      var css = 'appearance:button; -webkit-appearance:button; padding:' + padding + '; font-size:' + fontSize + '; font-family:' + font + '; cursor:pointer; border-radius:' + radius + '; border:' + border + '; background-color:' + (bg || '#f0f2f6') + '; color:' + (color || 'rgb(49,51,63)') + '; box-shadow:' + boxShadow + '; transition:background-color 0.15s ease;';
      var buttons = document.querySelectorAll('#dr-reset, #dr-save');
      for (var i = 0; i < buttons.length; i++) {{
        buttons[i].style.cssText = css;
      }}
      var label = document.getElementById('dr-color-label');
      if (label && color) label.style.color = color;
      if (label && font) label.style.fontFamily = font;
    }} catch (e) {{}}
  }}
  applyStreamlitStyles();
  document.getElementById('dr-reset').addEventListener('mouseenter', function() {{ this.style.filter = 'brightness(0.95)'; }});
  document.getElementById('dr-reset').addEventListener('mouseleave', function() {{ this.style.filter = ''; }});
  document.getElementById('dr-save').addEventListener('mouseenter', function() {{ this.style.filter = 'brightness(0.95)'; }});
  document.getElementById('dr-save').addEventListener('mouseleave', function() {{ this.style.filter = ''; }});

  var W = {w}, H = {h}, CH = {channels};
  var R = {r}, G = {g}, B = {b};
  var b64 = "{b64_esc}";
  var bin = atob(b64);
  var buf = new Uint8Array(bin.length);
  for (var i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);

  var canvas = document.getElementById('dr-canvas');
  var ctx = canvas.getContext('2d');
  var imgData = ctx.createImageData(W, H);

  function fillImageData() {{
    if (CH === 1) {{
      for (var i = 0; i < W * H; i++) {{
        var v = buf[i];
        imgData.data[i*4] = imgData.data[i*4+1] = imgData.data[i*4+2] = v;
        imgData.data[i*4+3] = 255;
      }}
    }} else if (CH === 4) {{
      for (var i = 0; i < W * H; i++) {{
        var j = i * 4;
        imgData.data[i*4] = buf[j];
        imgData.data[i*4+1] = buf[j+1];
        imgData.data[i*4+2] = buf[j+2];
        imgData.data[i*4+3] = buf[j+3];
      }}
    }} else {{
      for (var i = 0; i < W * H; i++) {{
        var j = i * 3;
        imgData.data[i*4] = buf[j];
        imgData.data[i*4+1] = buf[j+1];
        imgData.data[i*4+2] = buf[j+2];
        imgData.data[i*4+3] = 255;
      }}
    }}
    ctx.putImageData(imgData, 0, 0);
  }}
  fillImageData();

  document.getElementById('dr-color').addEventListener('input', function() {{
    var hex = this.value.replace('#','');
    R = parseInt(hex.substr(0,2), 16);
    G = parseInt(hex.substr(2,2), 16);
    B = parseInt(hex.substr(4,2), 16);
  }});

  document.getElementById('dr-reset').addEventListener('click', function() {{
    fillImageData();
  }});

  document.getElementById('dr-save').addEventListener('click', function() {{
    var link = document.createElement('a');
    link.download = 'image_with_regions.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
  }});

  function fillRect(x1, y1, x2, y2) {{
    for (var y = y1; y < y2; y++) {{
      for (var x = x1; x < x2; x++) {{
        var i = (y * W + x) * 4;
        imgData.data[i] = R;
        imgData.data[i+1] = G;
        imgData.data[i+2] = B;
        imgData.data[i+3] = 255;
      }}
    }}
    ctx.putImageData(imgData, 0, 0);
  }}

  var startX = null, startY = null;
  function toImgCoords(ev) {{
    var rect = canvas.getBoundingClientRect();
    var x = Math.floor((ev.clientX - rect.left) / rect.width * W);
    var y = Math.floor((ev.clientY - rect.top) / rect.height * H);
    return {{ x: Math.max(0, Math.min(W-1, x)), y: Math.max(0, Math.min(H-1, y)) }};
  }}
  canvas.addEventListener('mousedown', function(ev) {{
    var p = toImgCoords(ev);
    startX = p.x; startY = p.y;
  }});
  canvas.addEventListener('mousemove', function(ev) {{
    if (startX === null) return;
    var p = toImgCoords(ev);
    ctx.putImageData(imgData, 0, 0);
    ctx.strokeStyle = 'rgba(255,255,0,0.9)';
    ctx.lineWidth = 2;
    ctx.strokeRect(Math.min(startX,p.x), Math.min(startY,p.y), Math.abs(p.x-startX)+1, Math.abs(p.y-startY)+1);
  }});
  canvas.addEventListener('mouseup', function(ev) {{
    if (startX === null) return;
    var p = toImgCoords(ev);
    var x1 = Math.min(startX, p.x), x2 = Math.max(startX, p.x)+1;
    var y1 = Math.min(startY, p.y), y2 = Math.max(startY, p.y)+1;
    startX = null; startY = null;
    if (x2 <= x1 || y2 <= y1) return;
    fillRect(x1, y1, x2, y2);
  }});
  canvas.addEventListener('mouseleave', function() {{ startX = null; startY = null; ctx.putImageData(imgData, 0, 0); }});
}})();
</script>
"""


def _render_result_images(result, lab_choice, for_display):
    """Відмалювати блок результатів (зображення + опис лаб 3) у головній області. result — словник з images, algorithm, visual."""
    if not result:
        return
    SKIP_TITLE = {"Завантажене зображення", "Зображення", "Копія (збережено)"}
    images = result.get("images", [])
    if images:
        cols = st.columns(2)
        for i, (title, img) in enumerate(images):
            col = cols[i % 2]
            with col:
                if title not in SKIP_TITLE:
                    st.markdown(f"**{title}**")
                st.image(for_display(img), use_container_width=True)
                st.markdown("")
    if lab_choice in (3, 4) and (result.get("algorithm") or result.get("visual")):
        with st.expander("Опис алгоритму та візуальний ефект", expanded=True):
            if result.get("algorithm"):
                st.markdown("**Як влаштовано:**")
                st.caption(result["algorithm"])
            if result.get("visual"):
                st.markdown("**Візуально:**")
                st.caption(result["visual"])


def main():
    st.sidebar.title("Лабораторні з CV")
    st.sidebar.caption("Оберіть лабу, завдання та вхідні дані.")

    lab_choice = st.sidebar.selectbox(
        "Лабораторна",
        list(LAB_MODULES.keys()),
        format_func=lambda k: LAB_MODULES[k][1],
    )

    need_image = lab_choice != 5
    image_path = None
    image_from_upload = False

    if need_image:
        test_files = get_test_image_files()
        test_options = ["— не обрано —"] + test_files
        selected_test = st.sidebar.selectbox(
            "Тестове зображення",
            range(len(test_options)),
            format_func=lambda i: test_options[i],
        )
        uploaded = st.sidebar.file_uploader(
            "Або завантажити файл",
            type=["png", "jpg", "jpeg", "bmp"],
        )
        if uploaded:
            ext = os.path.splitext(uploaded.name)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
                f.write(uploaded.getvalue())
                f.flush()
                image_path = f.name
            image_from_upload = True
        elif selected_test > 0 and test_files:
            chosen = test_options[selected_test]
            image_path = os.path.join(TEST_IMAGES_DIR, chosen)
    else:
        st.sidebar.caption("Лаб 5 — вебкамера.")

    # ----- Лаб 1: кнопки та підписи в сайдбарі -----
    if lab_choice == 1:
        from core.io import load_image, get_dimensions, save_image
        from core.color import to_grayscale
        if "lab1_path" not in st.session_state:
            st.session_state.lab1_path = None
            st.session_state.lab1_original = None
            st.session_state.lab1_show_gray = False
        if image_path != st.session_state.lab1_path:
            st.session_state.lab1_path = image_path
            st.session_state.lab1_original = load_image(image_path) if image_path else None
            st.session_state.lab1_show_gray = False
        if need_image and image_path and st.session_state.lab1_original is not None:
            orig = st.session_state.lab1_original
            show_gray = st.session_state.lab1_show_gray
            current = to_grayscale(orig) if show_gray else orig
            h, w = get_dimensions(current)[:2]
            ch = f", {current.shape[2]} канали" if len(current.shape) >= 3 else ""
            st.sidebar.caption(f"{w} × {h} пікселів{ch}")
            if show_gray:
                if st.sidebar.button("Показати оригінал", use_container_width=True):
                    st.session_state.lab1_show_gray = False
                    st.rerun()
            else:
                if st.sidebar.button("Перетворити у grayscale", use_container_width=True):
                    st.session_state.lab1_show_gray = True
                    st.rerun()
            if st.sidebar.button("Зберегти", use_container_width=True):
                base, ext = os.path.splitext(image_path)
                out_path = base + ("_gray" if show_gray else "_copy") + ext
                if save_image(out_path, current):
                    st.sidebar.success(f"Збережено: {os.path.basename(out_path)}")
                else:
                    st.sidebar.caption("Помилка збереження")
            st.image(for_display(current), use_container_width=True)
        elif need_image and not image_path:
            st.sidebar.caption("Оберіть або завантажте зображення.")
        # Головна область — тільки зображення або порожня
        return

    # ----- Лаби 2–6: випадаючий список пунктів + одна кнопка -----
    mod = get_lab_module(lab_choice)
    tasks = mod.get_tasks()
    task_names = [t[0] for t in tasks]
    task_idx = st.sidebar.selectbox(
        "Пункт завдання",
        range(len(task_names)),
        format_func=lambda i: f"{i + 1}. {task_names[i]}",
    )
    st.sidebar.divider()
    current_task_name = task_names[task_idx]
    is_pixel_tracker = lab_choice == 2 and task_idx == 0
    is_draw_region = lab_choice == 2 and task_idx == 1
    is_bgr_hsv = lab_choice == 2 and task_idx == 2
    seg_color_hex = None
    lab3_kwargs = {}
    lab4_second_image = None
    if lab_choice == 2 and task_idx == 3:
        seg_color_hex = st.sidebar.color_picker("Колір для сегментації", value="#FF0000", key="lab2_seg_color")

    # Лаб 3: налаштування (зміна параметрів одразу застосовується)
    if lab_choice == 3:
        st.sidebar.subheader("Налаштування алгоритму")
        if task_idx in (0, 1, 2, 3):
            st.sidebar.slider("Розмір ядра", 3, 21, 5, 2, key="lab3_ksize")
            st.sidebar.caption("Розмір вікна згортки (тільки непарні). Чим більше — тим сильніше розмиття.")
        if task_idx == 1:
            st.sidebar.slider("Sigma (гаус)", 0.0, 10.0, 0.0, 0.5, key="lab3_sigma")
            st.sidebar.caption("Стандартне відхилення гаусіани по X; 0 = автоматично від розміру ядра.")
        if task_idx == 4:
            st.sidebar.caption("Порівняння blur з фіксованими ядрами 3×3, 5×5 та 9×9 — налаштувань немає.")
        if task_idx == 5:
            st.sidebar.slider("Розмір ядра Собеля/Лапласа", 1, 7, 3, 2, key="lab3_edge_ksize")
            st.sidebar.caption("Розмір ядра для обчислення похідних (1, 3, 5 або 7). Більше ядро — товстіші лінії країв.")
        if task_idx in (6, 7, 8, 9):
            st.sidebar.slider("Canny: нижній поріг", 0, 255, 50, key="lab3_canny_low")
            st.sidebar.caption("Нижній поріг гістерезису — слабкі краї нижче відкидаються.")
            st.sidebar.slider("Canny: верхній поріг", 0, 255, 150, key="lab3_canny_high")
            st.sidebar.caption("Верхній поріг гістерезису — сильні краї завжди зберігаються.")
        if task_idx == 10:
            st.sidebar.slider("Поріг", 0, 255, 127, key="lab3_thresh")
            st.sidebar.caption("Пікселі яскравіші за поріг стають білими (255), решта — чорними (0).")
        if task_idx in (11, 12, 13, 14):
            st.sidebar.slider("Поріг бінаризації", 0, 255, 127, key="lab3_morph_thresh")
            st.sidebar.caption("Поріг для отримання бінарного зображення перед морфологією.")
        if task_idx in (11, 12, 13):
            st.sidebar.slider("Розмір ядра морфології", 2, 15, 3, 1, key="lab3_morph_ksize")
            st.sidebar.caption("Розмір структурного елементу (квадрат); впливає на силу ерозії/дилатації.")
        if task_idx == 14:
            st.sidebar.caption("Очищення шуму: застосовується відкриття + закриття з фіксованим ядром.")
        lab3_kwargs = _lab3_kwargs_from_session(task_idx)

    # Лаб 4: налаштування (зміна параметрів одразу застосовується)
    lab4_kwargs = {}
    if lab_choice == 4:
        st.sidebar.subheader("Налаштування алгоритму")
        if task_idx == 0:
            st.sidebar.slider("Поріг", 0, 255, 127, key="lab4_thresh")
            st.sidebar.caption("Глобальний поріг для простого порогування.")
        if task_idx == 1:
            st.sidebar.slider("Розмір блоку", 3, 31, 11, 2, key="lab4_block_size")
            st.sidebar.caption("Розмір локального вікна для адаптивного порогу (непарне).")
            st.sidebar.slider("Зсув C", -10, 10, 2, key="lab4_C")
            st.sidebar.caption("Константа, що віднімається від середнього в блоці.")
        if task_idx == 2:
            st.sidebar.caption("Метод Оцу — поріг обирається автоматично, налаштувань немає.")
        if task_idx == 3:
            st.sidebar.slider("K (кількість кластерів)", 2, 16, 3, 1, key="lab4_K")
            st.sidebar.caption("Кількість кольорів у k-means сегментації.")
        if task_idx == 4:
            st.sidebar.caption("Порівняння методів — фіксовані параметри.")
        if task_idx == 5:
            st.sidebar.slider("Harris: block_size", 2, 10, 2, 1, key="lab4_harris_block")
            st.sidebar.slider("Harris: ksize", 3, 15, 3, 2, key="lab4_harris_ksize")
            st.sidebar.slider("Harris: k", 0.01, 0.2, 0.04, 0.01, key="lab4_harris_k")
            st.sidebar.caption("Параметри детектора кутів Harris.")
        if task_idx in (6, 7, 8):
            st.sidebar.caption("ORB та матчинг — параметри за замовчуванням.")
        # Пункт 9 (task_idx 8): друге зображення для матчингу між двома фото (необов'язково)
        if task_idx == 8:
            st.sidebar.caption("Друге зображення (необов'язково): для матчингу між двома різними фото.")
            test_options_2 = ["— не обрано —"] + get_test_image_files()
            selected_2 = st.sidebar.selectbox(
                "Друге тестове зображення",
                range(len(test_options_2)),
                format_func=lambda i: test_options_2[i],
                key="lab4_second_select",
            )
            uploaded_2 = st.sidebar.file_uploader(
                "Або завантажити друге фото",
                type=["png", "jpg", "jpeg", "bmp"],
                key="lab4_second_upload",
            )
            if uploaded_2:
                ext2 = os.path.splitext(uploaded_2.name)[1] or ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext2) as f2:
                    f2.write(uploaded_2.getvalue())
                    f2.flush()
                    lab4_second_image = f2.name
            elif selected_2 > 0 and len(test_options_2) > 1:
                lab4_second_image = os.path.join(TEST_IMAGES_DIR, test_options_2[selected_2])
        lab4_kwargs = _lab4_kwargs_from_session(task_idx)
        if task_idx == 8 and lab4_second_image:
            lab4_kwargs["image_path_2"] = lab4_second_image

    # Лаб 5: тривалість живого перегляду (п. 1–4), ROI для п. 4, тривалість запису для п. 5
    lab5_kwargs = {}
    if lab_choice == 5:
        st.sidebar.subheader("Відеопотік")
        st.sidebar.caption("Потік йде поки відкритий пункт. Щоб зупинити — оновіть сторінку або змініть пункт.")
        if task_idx == 1:
            st.sidebar.slider("Розмір ядра розмиття", 3, 21, 5, 2, key="lab5_blur_ksize")
            st.sidebar.caption("Непарне число; більше — сильніше розмиття.")
            st.sidebar.slider("Sigma (гаус)", 0.0, 10.0, 0.0, 0.5, key="lab5_blur_sigma")
            st.sidebar.caption("0 — автоматично від розміру ядра.")
        if task_idx == 3:
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                w_cam = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                h_cam = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                cap.release()
            else:
                w_cam, h_cam = 640, 480
            st.sidebar.caption(f"Роздільність камери: {w_cam} × {h_cam}")
            st.sidebar.slider("ROI x1", 0, max(1, w_cam - 1), min(100, w_cam - 1), key="lab5_roi_x1")
            st.sidebar.slider("ROI y1", 0, max(1, h_cam - 1), min(100, h_cam - 1), key="lab5_roi_y1")
            st.sidebar.slider("ROI x2", 0, w_cam, min(300, w_cam), key="lab5_roi_x2")
            st.sidebar.slider("ROI y2", 0, h_cam, min(300, h_cam), key="lab5_roi_y2")
        if task_idx == 4:
            st.sidebar.slider("Тривалість запису (с)", 1, 10, 2, 1, key="lab5_record_sec")
            if st.sidebar.button("Запустити запис", type="primary", use_container_width=True):
                st.session_state.lab5_start_record = True
                st.rerun()

    # Лаб 6: кольорова сегментація — діапазони HSV та очищення морфологією
    lab6_kwargs = {}
    if lab_choice == 6:
        st.sidebar.subheader("Налаштування сегментації")
        if task_idx == 0:
            st.sidebar.slider("H (відтінок): мін", 0, 180, 0, key="lab6_h_low")
            st.sidebar.slider("H (відтінок): макс", 0, 180, 30, key="lab6_h_high")
            st.sidebar.slider("S (насиченість): мін", 0, 255, 50, key="lab6_s_low")
            st.sidebar.slider("S (насиченість): макс", 0, 255, 255, key="lab6_s_high")
            st.sidebar.slider("V (яскравість): мін", 0, 255, 50, key="lab6_v_low")
            st.sidebar.slider("V (яскравість): макс", 0, 255, 255, key="lab6_v_high")
            st.sidebar.checkbox("Очищення маски морфологією", value=True, key="lab6_denoise")
            st.sidebar.caption("Діапазон HSV для виділення кольору; морфологія зменшує шум на маску.")
        else:
            st.sidebar.checkbox("Метод Оцу (автопоріг)", value=True, key="lab6_use_otsu")
            st.sidebar.slider("Поріг (якщо не Оцу)", 0, 255, 127, key="lab6_count_thresh")
            st.sidebar.slider("Мін. площа об'єкта (пікселів)", 10, 5000, 100, 10, key="lab6_min_area")
            st.sidebar.caption("Контури з площею менше мін. відкидаються (шум).")

    if is_pixel_tracker:
        run_clicked = False
        st.sidebar.caption("Наведіть курсор на зображення — значення пікселя з’явиться під ним.")
    elif is_draw_region:
        run_clicked = False
    elif lab_choice == 5 and task_idx == 4:
        run_clicked = False
    else:
        # Без кнопки: зміни параметрів одразу застосовуються (onChange)
        run_clicked = True

    # Якщо зображення обрано
    if need_image and image_path:
        from core.io import load_image, get_dimensions
        img_loaded = load_image(image_path)
        if img_loaded is not None:
            h, w = get_dimensions(img_loaded)[:2]
            ch = f", {img_loaded.shape[2]} канали" if len(img_loaded.shape) >= 3 else ""
            st.sidebar.caption(f"{w} × {h} пікселів{ch}")
            # Режим «значення пікселя»: без кнопки, трекер миші в реальному часі
            if is_pixel_tracker:
                img = np.ascontiguousarray(img_loaded)
                h_img, w_img = img.shape[:2]
                chans = 1 if len(img.shape) == 2 else img.shape[2]
                raw = img[:, :, :chans].tobytes() if len(img.shape) == 3 else img.tobytes()
                b64 = base64.b64encode(raw).decode("ascii")
                b64_esc = b64.replace("\\", "\\\\").replace('"', '\\"')
                html = _pixel_tracker_html(w_img, h_img, chans, b64_esc)
                components.html(html, height=h_img + 80, scrolling=False)
                return
            # Режим «змінити область»: колір у сайдбарі, прямокутники мишкою, кнопка «Зберегти» під зображенням
            # Використовуємо components.html з явною висотою, щоб зображення завантажувалось повністю (iframe-event обрізав через cross-origin).
            if is_draw_region:
                draw_img = img_loaded
                display_img = for_display(draw_img)
                chans = 1 if len(display_img.shape) == 2 else display_img.shape[2]
                raw = display_img[:, :, :chans].tobytes() if len(display_img.shape) == 3 else display_img.tobytes()
                b64 = base64.b64encode(raw).decode("ascii")
                b64_esc = b64.replace("\\", "\\\\").replace('"', '\\"')
                html = _draw_rectangles_html(w, h, chans, b64_esc, 0, 255, 0, 0)
                components.html(html, height=h + 100, scrolling=False)
                return
            if is_bgr_hsv:
                result = run_task_with_image(lab_choice, task_idx, image_path)
                if result.get("text"):
                    st.sidebar.caption(result["text"])
                images = result.get("images", [])
                if images:
                    cols = st.columns(2)
                    for i, (title, img) in enumerate(images):
                        with cols[i % 2]:
                            st.markdown(f"**{title}**")
                            st.image(img, use_container_width=True)
                return
            # Інші лаби: не показуємо попередній перегляд, одразу переходимо до виконання нижче

    # Показувати результат тільки коли є вхідні дані (зображення для лаб 1–4, 6 або без для лаб 5)
    do_run_lab5_record = lab_choice == 5 and task_idx == 4 and st.session_state.pop("lab5_start_record", False)
    if not run_clicked and not do_run_lab5_record:
        return
    if need_image and not image_path:
        st.sidebar.warning("Оберіть зображення або завантажте своє.")
        return

    # Плейсхолдер: попередній результат тільки для того самого пункту; при зміні пункту — одразу прибираємо
    result_placeholder = st.empty()
    result_key = (
        lab_choice,
        task_idx,
        image_path if need_image else None,
        lab4_second_image if (lab_choice == 4 and task_idx == 8) else None,
    )
    last_key = st.session_state.get("lab_last_result_key")
    last_result = st.session_state.get("lab_last_result") if last_key == result_key else None
    with result_placeholder.container():
        if last_result:
            _render_result_images(last_result, lab_choice, for_display)

    with st.spinner("Виконується…"):
        try:
            kwargs = {}
            if lab_choice == 2 and task_idx == 3 and seg_color_hex:
                h = seg_color_hex.lstrip("#")
                r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                kwargs["target_color_bgr"] = (b, g, r)
            if lab_choice == 3 and lab3_kwargs:
                kwargs.update(lab3_kwargs)
            if lab_choice == 4 and lab4_kwargs:
                kwargs.update(lab4_kwargs)
            if lab_choice == 6:
                if task_idx == 0:
                    kwargs["h_low"] = st.session_state.get("lab6_h_low", 0)
                    kwargs["h_high"] = st.session_state.get("lab6_h_high", 30)
                    kwargs["s_low"] = st.session_state.get("lab6_s_low", 50)
                    kwargs["s_high"] = st.session_state.get("lab6_s_high", 255)
                    kwargs["v_low"] = st.session_state.get("lab6_v_low", 50)
                    kwargs["v_high"] = st.session_state.get("lab6_v_high", 255)
                    kwargs["denoise"] = st.session_state.get("lab6_denoise", True)
                else:
                    kwargs["use_otsu"] = st.session_state.get("lab6_use_otsu", True)
                    kwargs["thresh"] = st.session_state.get("lab6_count_thresh", 127)
                    kwargs["min_area"] = st.session_state.get("lab6_min_area", 100)
            if lab_choice == 5:
                stream_placeholder = st.empty()
                # Без обмеження по часу: потік йде поки відкритий пункт; при зміні пункту — перезавантаження
                duration_sec = 3600.0  # 1 година; зупинка — оновити сторінку або змінити пункт
                kwargs["device"] = 0
                if task_idx == 0:
                    kwargs["duration_sec"] = duration_sec
                    kwargs["record"] = False
                    kwargs["stream_callback"] = lambda f: stream_placeholder.image(for_display(f), caption="Відеопотік", use_container_width=True)
                elif task_idx == 1:
                    kwargs["duration_sec"] = duration_sec
                    kwargs["blur_ksize"] = st.session_state.get("lab5_blur_ksize", 5)
                    kwargs["blur_sigma"] = st.session_state.get("lab5_blur_sigma", 0.0)
                    def _lab5_cb_2(frame, gray, blurred):
                        gray_bgr = gray_to_bgr(gray) if gray.ndim == 2 else gray
                        combined = np.hstack([frame, gray_bgr, blurred])
                        stream_placeholder.image(for_display(combined), caption="Оригінал | Grayscale | Blur", use_container_width=True)
                    kwargs["stream_callback"] = _lab5_cb_2
                elif task_idx == 2:
                    kwargs["duration_sec"] = duration_sec
                    def _lab5_cb_3(frame, diff, thresh):
                        diff_bgr = gray_to_bgr(diff)
                        thresh_bgr = gray_to_bgr(thresh)
                        combined = np.hstack([frame, diff_bgr, thresh_bgr])
                        stream_placeholder.image(for_display(combined), caption="Кадр | Різниця (рух) | Поріг", use_container_width=True)
                    kwargs["stream_callback"] = _lab5_cb_3
                elif task_idx == 3:
                    kwargs["duration_sec"] = duration_sec
                    kwargs["x1"] = st.session_state.get("lab5_roi_x1", 100)
                    kwargs["y1"] = st.session_state.get("lab5_roi_y1", 100)
                    kwargs["x2"] = st.session_state.get("lab5_roi_x2", 300)
                    kwargs["y2"] = st.session_state.get("lab5_roi_y2", 300)
                    def _lab5_cb_4(frame_roi, roi):
                        h_f, w_f = frame_roi.shape[:2]
                        h_r, w_r = roi.shape[:2]
                        right = np.zeros((h_f, w_r) + frame_roi.shape[2:], dtype=frame_roi.dtype)
                        y0 = (h_f - h_r) // 2
                        right[y0 : y0 + h_r, :] = roi
                        combined = np.hstack([frame_roi, right])
                        stream_placeholder.image(for_display(combined), caption="Кадр з ROI | Вирізаний ROI", use_container_width=True)
                    kwargs["stream_callback"] = _lab5_cb_4
                elif task_idx == 4:
                    kwargs["num_seconds"] = st.session_state.get("lab5_record_sec", 2)
                    kwargs["stream_callback"] = lambda f: stream_placeholder.image(for_display(f), caption="Запис відео", use_container_width=True)
            result = run_task_with_image(lab_choice, task_idx, image_path if need_image else None, **kwargs)
        except Exception as e:
            st.error(f"Помилка: {e}")
            return
        finally:
            if need_image and image_path and image_from_upload and os.path.exists(image_path):
                try:
                    os.unlink(image_path)
                except Exception:
                    pass

    st.session_state["lab_last_result"] = result
    st.session_state["lab_last_result_key"] = result_key

    # Текст результату — у сайдбарі
    if result.get("text"):
        st.sidebar.caption(result["text"])

    # Оновлюємо плейсхолдер новим результатом (без миготіння — той самий блок контенту)
    with result_placeholder.container():
        _render_result_images(result, lab_choice, for_display)

    if result.get("video_path") and os.path.exists(result["video_path"]):
        try:
            with open(result["video_path"], "rb") as f:
                video_bytes = f.read()
            if video_bytes:
                st.video(video_bytes, format="video/mp4")
            else:
                st.caption("Відео порожнє або не вдалося прочитати.")
        except Exception as e:
            st.error(f"Не вдалося відтворити відео: {e}")


if __name__ == "__main__":
    main()
