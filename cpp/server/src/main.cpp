#include <gst/gst.h>
#include <gst/webrtc/webrtc.h>
#include <glib.h>
#include <json-glib/json-glib.h>
#include <libsoup/soup.h>

static GMainLoop *loop;
static GstElement *pipeline, *webrtc;
static SoupWebsocketConnection *ws_conn = NULL;

static void on_offer_created(GstPromise *promise, gpointer user_data);
static void on_negotiation_needed(GstElement *element, gpointer user_data);
static void on_ice_candidate(GstElement *element, guint mline_index, gchar *candidate, gpointer user_data);
static void send_sdp_offer(GstWebRTCSessionDescription *offer);
static void send_ice_candidate_message(guint mline_index, const gchar *candidate);
static void on_server_message(SoupWebsocketConnection *conn, SoupWebsocketDataType type, GBytes *message, gpointer user_data);
static void on_websocket_connected(SoupSession *session, GAsyncResult *res, gpointer user_data);

static void on_offer_created(GstPromise *promise, gpointer user_data)
{
  GstWebRTCSessionDescription *offer = NULL;
  const GstStructure *reply = gst_promise_get_reply(promise);
  gst_structure_get(reply, "offer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &offer, NULL);
  gst_promise_unref(promise);

  GstPromise *local_promise = gst_promise_new();
  g_signal_emit_by_name(webrtc, "set-local-description", offer, local_promise);
  gst_promise_interrupt(local_promise);
  gst_promise_unref(local_promise);

  send_sdp_offer(offer);
  gst_webrtc_session_description_free(offer);
}

static void on_negotiation_needed(GstElement *element, gpointer user_data)
{
  GstPromise *promise = gst_promise_new_with_change_func(on_offer_created, user_data, NULL);
  g_signal_emit_by_name(webrtc, "create-offer", NULL, promise);
}

static void on_ice_candidate(GstElement *element, guint mline_index, gchar *candidate, gpointer user_data)
{
  send_ice_candidate_message(mline_index, candidate);
}

static void send_sdp_offer(GstWebRTCSessionDescription *offer)
{
  gchar *sdp_str = gst_sdp_message_as_text(offer->sdp);
  JsonObject *message = json_object_new();
  json_object_set_string_member(message, "type", "offer");
  json_object_set_string_member(message, "sdp", sdp_str);

  gchar *text = json_to_string(json_node_init_object(json_node_alloc(), message), FALSE);
  soup_websocket_connection_send_text(ws_conn, text);
  g_free(text);
  g_free(sdp_str);
  json_object_unref(message);
}

static void send_ice_candidate_message(guint mline_index, const gchar *candidate)
{
  JsonObject *message = json_object_new();
  json_object_set_string_member(message, "type", "ice-candidate");
  json_object_set_int_member(message, "sdpMLineIndex", mline_index);
  json_object_set_string_member(message, "candidate", candidate);

  gchar *text = json_to_string(json_node_init_object(json_node_alloc(), message), FALSE);
  soup_websocket_connection_send_text(ws_conn, text);
  g_free(text);
  json_object_unref(message);
}

static void on_server_message(SoupWebsocketConnection *conn, SoupWebsocketDataType type, GBytes *message, gpointer user_data)
{
  gsize size;
  gchar *text = (gchar *)g_bytes_unref_to_data(message, &size);
  JsonParser *json_parser = json_parser_new();
  if (!json_parser_load_from_data(json_parser, text, size, NULL))
  {
    g_error("Failed to parse JSON message");
    g_free(text);
    return;
  }

  JsonObject *root = json_node_get_object(json_parser_get_root(json_parser));
  const gchar *type_str = json_object_get_string_member(root, "type");

  if (g_strcmp0(type_str, "answer") == 0)
  {
    const gchar *sdp_str = json_object_get_string_member(root, "sdp");
    GstSDPMessage *sdp;
    gst_sdp_message_new(&sdp);
    gst_sdp_message_parse_buffer((guint8 *)sdp_str, strlen(sdp_str), sdp);

    GstWebRTCSessionDescription *answer = gst_webrtc_session_description_new(GST_WEBRTC_SDP_TYPE_ANSWER, sdp);
    GstPromise *promise = gst_promise_new();
    g_signal_emit_by_name(webrtc, "set-remote-description", answer, promise);
    gst_promise_interrupt(promise);
    gst_promise_unref(promise);
    gst_webrtc_session_description_free(answer);
  }
  else if (g_strcmp0(type_str, "ice-candidate") == 0)
  {
    guint mline_index = json_object_get_int_member(root, "sdpMLineIndex");
    const gchar *candidate = json_object_get_string_member(root, "candidate");
    g_signal_emit_by_name(webrtc, "add-ice-candidate", mline_index, candidate);
  }

  g_object_unref(json_parser);
  g_free(text);
}

static void on_websocket_connected(SoupSession *session, GAsyncResult *res, gpointer user_data)
{
  GError *error = NULL;
  ws_conn = soup_session_websocket_connect_finish(session, res, &error);
  if (error)
  {
    g_error("Failed to connect to WebSocket server: %s", error->message);
    g_error_free(error);
    return;
  }

  g_signal_connect(ws_conn, "message", G_CALLBACK(on_server_message), NULL);
}

static void connect_to_signaling_server(void)
{
  SoupSession *session = soup_session_new();
  SoupMessage *msg = soup_message_new(SOUP_METHOD_GET, "ws://localhost:8443");

  soup_session_websocket_connect_async(session, msg, NULL, NULL, NULL, on_websocket_connected, NULL);
}

int main(int argc, char *argv[])
{
  gst_init(&argc, &argv);
  loop = g_main_loop_new(NULL, FALSE);

  pipeline = gst_pipeline_new("webrtc-send-pipeline");
  GstElement *src = gst_element_factory_make("v4l2src", "video-src");
  GstElement *videoconvert = gst_element_factory_make("videoconvert", "convert");
  GstElement *enc = gst_element_factory_make("vp8enc", "encoder");
  GstElement *pay = gst_element_factory_make("rtpvp8pay", "payloader");
  webrtc = gst_element_factory_make("webrtcbin", "webrtc");

  if (!pipeline || !src || !videoconvert || !enc || !pay || !webrtc)
  {
    g_error("Failed to create elements");
    return -1;
  }

  gst_bin_add_many(GST_BIN(pipeline), src, videoconvert, enc, pay, webrtc, NULL);

  if (!gst_element_link_many(src, videoconvert, enc, pay, webrtc, NULL))
  {
    g_error("Failed to link elements");
    return -1;
  }

  connect_to_signaling_server();

  g_signal_connect(webrtc, "on-negotiation-needed", G_CALLBACK(on_negotiation_needed), NULL);
  g_signal_connect(webrtc, "on-ice-candidate", G_CALLBACK(on_ice_candidate), NULL);

  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  g_main_loop_run(loop);

  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
  g_main_loop_unref(loop);

  return 0;
}
